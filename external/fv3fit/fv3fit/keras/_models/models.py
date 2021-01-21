from typing import Sequence, Tuple, Iterable, Mapping, Union, Optional, Any, List
from typing_extensions import Literal
import xarray as xr
import logging
import abc
import tensorflow as tf
import tensorflow_addons as tfa
from ..._shared import ArrayPacker, Estimator, io, unpack_matrix
import numpy as np
import os
from ._filesystem import get_dir, put_dir
from ._sequences import _XyArraySequence, _ThreadedSequencePreLoader
from .normalizer import LayerStandardScaler
from .loss import get_weighted_mse, get_weighted_mae
import yaml

logger = logging.getLogger(__file__)

MODEL_DIRECTORY = "model_data"

# Description of the training loss progression over epochs
# Outer array indexes epoch, inner array indexes batch (if applicable)
EpochLossHistory = Sequence[Sequence[Union[float, int]]]
History = Mapping[str, EpochLossHistory]


class PackedKerasModel(Estimator):
    """
    Abstract base class for a keras-based model which operates on xarray
    datasets containing a "sample" dimension (as defined by loaders.SAMPLE_DIM_NAME),
    where each variable has at most one non-sample dimension.

    Subclasses are defined primarily using a `get_model` method, which returns a
    Keras model.
    """

    # these should only be used in the dump/load routines for this class
    _MODEL_FILENAME = "model.tf"
    _X_PACKER_FILENAME = "X_packer.json"
    _Y_PACKER_FILENAME = "y_packer.json"
    _X_SCALER_FILENAME = "X_scaler.npz"
    _Y_SCALER_FILENAME = "y_scaler.npz"
    _OPTIONS_FILENAME = "options.yml"
    _LOSS_OPTIONS = {"mse": get_weighted_mse, "mae": get_weighted_mae}
    _HISTORY_FILENAME = "training_history.json"

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        weights: Optional[Mapping[str, Union[int, float, np.ndarray]]] = None,
        normalize_loss: bool = True,
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam,
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        loss: Literal["mse", "mae"] = "mse",
        checkpoint_path: Optional[str] = None,
    ):
        """Initialize the model.
        
        Loss is computed on normalized outputs only if `normalized_loss` is True
        (default). This allows you to provide weights that will be proportional
        to the importance of that feature within the loss. If `normalized_loss`
        is False, you should consider scaling your weights to decrease the importance
        of features that are orders of magnitude larger than other features.
        
        Args:
            sample_dim_name: name of the sample dimension in datasets used as
                inputs and outputs.
            input_variables: names of input variables
            output_variables: names of output variables
            weights: loss function weights, defined as a dict whose keys are
                variable names and values are either a scalar referring to the total
                weight of the variable, or a vector referring to the weight for each
                feature of the variable. Default is a total weight of 1
                for each variable.
            normalize_loss: if True (default), normalize outputs by their standard
                deviation before computing the loss function
            optimizer: algorithm to be used in gradient descent, must subclass
                tf.keras.optimizers.Optimizer; defaults to tf.keras.optimizers.Adam
            loss: loss function to use. Defaults to mean squared error.
        """
        super().__init__(sample_dim_name, input_variables, output_variables)
        self._model = None
        self.X_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, pack_names=input_variables
        )
        self.y_packer = ArrayPacker(
            sample_dim_name=sample_dim_name, pack_names=output_variables
        )
        self.X_scaler = LayerStandardScaler()
        self.y_scaler = LayerStandardScaler()
        self.train_history = {"loss": [], "val_loss": []}  # type: Mapping[str, List]
        if weights is None:
            self.weights: Mapping[str, Union[int, float, np.ndarray]] = {}
        else:
            self.weights = weights
        self._normalize_loss = normalize_loss
        self._optimizer = optimizer
        self._loss = loss
        self._kernel_regularizer = kernel_regularizer
        self._checkpoint_path = checkpoint_path

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("must call fit() for keras model to be available")
        return self._model

    def _fit_normalization(self, X: np.ndarray, y: np.ndarray):
        self.X_scaler.fit(X)
        self.y_scaler.fit(y)

    @abc.abstractmethod
    def get_model(self, n_features_in: int, n_features_out: int) -> tf.keras.Model:
        """Returns a Keras model to use as the underlying predictive model.
        
        Args:
            n_features_in: the number of input features
            n_features_out: the number of output features
        Returns:
            model: a Keras model whose input shape is [n_samples, n_features_in] and
                output shape is [n_samples, features_out]
        """

    def fit(
        self,
        batches: Sequence[xr.Dataset],
        validation_dataset: Optional[xr.Dataset] = None,
        epochs: int = 1,
        batch_size: Optional[int] = None,
        workers: int = 1,
        max_queue_size: int = 8,
        validation_samples: int = 13824,
        **fit_kwargs: Any,
    ) -> None:
        """Fits a model using data in the batches sequence
        
        If batch_size is provided as a kwarg, the list of values is for each batch fit.
        e.g. {"loss":
            [[epoch0_batch0_loss, epoch0_batch1_loss],
            [epoch1_batch0_loss, epoch1_batch1_loss]]}
        If not batch_size is not provided, a single loss per epoch is recorded.
        e.g. {"loss": [[epoch0_loss], [epoch1_loss]]}
        
        Args:
            batches: sequence of stacked datasets of predictor variables
            validation_dataset: optional validation dataset
            epochs: optional number of times through the batches to run when training
            batch_size: actual batch_size to apply in gradient descent updates,
                independent of number of samples in each batch in batches; optional,
                uses number of samples in each batch if omitted
            workers: number of workers for parallelized loading of batches fed into
                training, defaults to serial loading (1 worker)
            max_queue_size: max number of batches to hold in the parallel loading queue
            validation_samples: Option to specify number of samples to randomly draw
                from the validation dataset, so that we can use multiple timesteps for
                validation without having to load all the times into memory.
                Defaults to the equivalent of a single C48 timestep.
            **fit_kwargs: other keyword arguments to be passed to the underlying
                tf.keras.Model.fit() method
        """
        epochs = epochs if epochs is not None else 1
        Xy = _XyArraySequence(self.X_packer, self.y_packer, batches)

        if self._model is None:
            X, y = Xy[0]
            n_features_in, n_features_out = X.shape[-1], y.shape[-1]
            self._fit_normalization(X, y)
            self._model = self.get_model(n_features_in, n_features_out)

        validation_data: Optional[Tuple[np.ndarray, np.ndarray]]
        if validation_dataset is not None:
            X_val = self.X_packer.to_array(validation_dataset)
            y_val = self.y_packer.to_array(validation_dataset)
            val_sample = np.random.choice(
                np.arange(len(y_val)), validation_samples, replace=False
            )
            validation_data = X_val[val_sample], y_val[val_sample]
        else:
            validation_data = None

        return self._fit_loop(
            Xy,
            validation_data,
            epochs,
            batch_size,
            workers=workers,
            max_queue_size=max_queue_size,
            **fit_kwargs,
        )

    def _fit_loop(
        self,
        Xy: Sequence[Tuple[np.ndarray, np.ndarray]],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
        epochs: int,
        batch_size: Optional[int] = None,
        workers: int = 1,
        max_queue_size: int = 8,
        **fit_kwargs: Any,
    ) -> None:

        if workers > 1:
            Xy = _ThreadedSequencePreLoader(
                Xy, num_workers=workers, max_queue_size=max_queue_size
            )

        for i_epoch in range(epochs):
            loss_over_batches, val_loss_over_batches = [], []
            for i_batch, (X, y) in enumerate(Xy):
                logger.info(
                    f"Fitting on timestep {i_batch} of {len(Xy)}, of epoch {i_epoch}..."
                )
                history = self.model.fit(
                    X,
                    y,
                    validation_data=validation_data,
                    batch_size=batch_size,
                    **fit_kwargs,
                )
                loss_over_batches += history.history["loss"]
                val_loss_over_batches += history.history.get("val_loss", [np.nan])
            self.train_history["loss"].append(loss_over_batches)
            self.train_history["val_loss"].append(val_loss_over_batches)
            if self._checkpoint_path:
                self.dump(os.path.join(self._checkpoint_path, f"epoch_{i_epoch}"))
                logger.info(
                    f"Saved model checkpoint after epoch {i_epoch} "
                    f"to {self._checkpoint_path}"
                )

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        sample_coord = X[self.sample_dim_name]
        ds_pred = self.y_packer.to_dataset(
            self.predict_array(self.X_packer.to_array(X))
        )
        return ds_pred.assign_coords({self.sample_dim_name: sample_coord})

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def dump(self, path: str) -> None:
        dir_ = os.path.join(path, MODEL_DIRECTORY)
        with put_dir(dir_) as path:
            if self._model is not None:
                model_filename = os.path.join(path, self._MODEL_FILENAME)
                self.model.save(model_filename)
            with open(os.path.join(path, self._X_PACKER_FILENAME), "w") as f:
                self.X_packer.dump(f)
            with open(os.path.join(path, self._Y_PACKER_FILENAME), "w") as f:
                self.y_packer.dump(f)
            with open(os.path.join(path, self._X_SCALER_FILENAME), "wb") as f_binary:
                self.X_scaler.dump(f_binary)
            with open(os.path.join(path, self._Y_SCALER_FILENAME), "wb") as f_binary:
                self.y_scaler.dump(f_binary)
            with open(os.path.join(path, self._OPTIONS_FILENAME), "w") as f:
                yaml.safe_dump(
                    {"normalize_loss": self._normalize_loss, "loss": self._loss}, f
                )
            with open(os.path.join(path, self._HISTORY_FILENAME), "w") as f:
                yaml.safe_dump(self.train_history, f)

    @property
    def loss(self):
        # putting this on a property method is needed so we can save and load models
        # using custom loss functions. If using a custom function, it must either
        # be named "custom_loss", as used in the load method below,
        # or it must be registered with keras as a custom object.
        # Do this by defining the function returned by the decorator as custom_loss.
        # See https://github.com/keras-team/keras/issues/5916 for more info
        std = self.y_scaler.std
        std[std == 0] = 1.0
        if not self._normalize_loss:
            std[:] = 1.0
        if self._loss in self._LOSS_OPTIONS:
            loss_getter = self._LOSS_OPTIONS[self._loss]
            return loss_getter(self.y_packer, std, **self.weights)
        else:
            raise ValueError(
                f"Invalid loss {self._loss} provided. "
                f"Allowed loss functions are {list(self._LOSS_OPTIONS.keys())}."
            )

    @classmethod
    def load(cls, path: str) -> "PackedKerasModel":
        dir_ = os.path.join(path, MODEL_DIRECTORY)
        with get_dir(dir_) as path:
            with open(os.path.join(path, cls._X_PACKER_FILENAME), "r") as f:
                X_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._Y_PACKER_FILENAME), "r") as f:
                y_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._X_SCALER_FILENAME), "rb") as f_binary:
                X_scaler = LayerStandardScaler.load(f_binary)
            with open(os.path.join(path, cls._Y_SCALER_FILENAME), "rb") as f_binary:
                y_scaler = LayerStandardScaler.load(f_binary)
            with open(os.path.join(path, cls._OPTIONS_FILENAME), "r") as f:
                options = yaml.safe_load(f)
                    
            obj = cls(
                X_packer.sample_dim_name,
                X_packer.pack_names,
                y_packer.pack_names,
                **options,
            )
            obj.X_packer = X_packer
            obj.y_packer = y_packer
            obj.X_scaler = X_scaler
            obj.y_scaler = y_scaler
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            if os.path.exists(model_filename):
                obj._model = tf.keras.models.load_model(
                    model_filename, custom_objects={"custom_loss": obj.loss}
                )
            return obj

    def jacobian(self, base_state: Optional[xr.Dataset] = None) -> xr.Dataset:
        """Compute the jacobian of the NN around a base state

        Args:
            base_state: a single sample of input data. If not passed, then
                the mean of the input data stored in the X_scaler will be used.

        Returns:
            The jacobian matrix as a Dataset

        """
        if base_state is None:
            if self.X_scaler.mean is not None:
                mean_expanded = self.X_packer.to_dataset(
                    self.X_scaler.mean[np.newaxis, :]
                )
            else:
                raise ValueError("X_scaler needs to be fit first.")
        else:
            mean_expanded = base_state.expand_dims(self.sample_dim_name)

        mean_tf = tf.convert_to_tensor(self.X_packer.to_array(mean_expanded))
        with tf.GradientTape() as g:
            g.watch(mean_tf)
            y = self.model(mean_tf)

        J = g.jacobian(y, mean_tf)[0, :, 0, :].numpy()
        return unpack_matrix(self.X_packer, self.y_packer, J)


@io.register("packed-keras")
class DenseModel(PackedKerasModel):
    """
    A simple feedforward neural network model with dense layers.
    """

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        weights: Optional[Mapping[str, Union[int, float, np.ndarray]]] = None,
        normalize_loss: bool = True,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        depth: int = 3,
        width: int = 16,
        loss: Literal["mse", "mae"] = "mse",
        spectral_normalization: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        """Initialize the DenseModel.

        Loss is computed on normalized outputs only if `normalized_loss` is True
        (default). This allows you to provide weights that will be proportional
        to the importance of that feature within the loss. If `normalized_loss`
        is False, you should consider scaling your weights to decrease the importance
        of features that are orders of magnitude larger than other features.

        Args:
            sample_dim_name: name of the sample dimension in datasets used as
                inputs and outputs.
            input_variables: names of input variables
            output_variables: names of output variables
            weights: loss function weights, defined as a dict whose keys are
                variable names and values are either a scalar referring to the total
                weight of the variable, or a vector referring to the weight for each
                feature of the variable. Default is a total weight of 1
                for each variable.
            normalize_loss: if True (default), normalize outputs by their standard
                deviation before computing the loss function
            optimizer: algorithm to be used in gradient descent, must subclass
                tf.keras.optimizers.Optimizer; defaults to tf.keras.optimizers.Adam
            depth: number of dense layers to use between the input and output layer.
                The number of hidden layers will be (depth - 1). Default is 3.
            width: number of neurons to use on layers between the input and output
                layer. Default is 16.
            loss: loss function to use. Defaults to mean squared error.
        """
        self._depth = depth
        self._width = width
        self._spectral_normalization = spectral_normalization
        optimizer = optimizer or tf.keras.optimizers.Adam()
        super().__init__(
            sample_dim_name,
            input_variables,
            output_variables,
            weights=weights,
            normalize_loss=normalize_loss,
            optimizer=optimizer,
            kernel_regularizer=kernel_regularizer,
            loss=loss,
            checkpoint_path=checkpoint_path,
        )

    def get_model(self, n_features_in: int, n_features_out: int) -> tf.keras.Model:
        inputs = tf.keras.Input(n_features_in)
        x = self.X_scaler.normalize_layer(inputs)
        for i in range(self._depth - 1):
            hidden_layer = tf.keras.layers.Dense(
                self._width,
                activation=tf.keras.activations.relu,
                kernel_regularizer=self._kernel_regularizer,
            )
            if self._spectral_normalization:
                hidden_layer = tfa.layers.SpectralNormalization(hidden_layer)
            x = hidden_layer(x)
        x = tf.keras.layers.Dense(n_features_out)(x)
        outputs = self.y_scaler.denormalize_layer(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self._optimizer, loss=self.loss)
        return model
