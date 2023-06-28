from fv3fit.reservoir.domain import RankDivider
from fv3fit.reservoir.readout import ReservoirComputingReadout
import numpy as np
from scipy import sparse

from fv3fit.reservoir import (
    ReservoirComputingModel,
    Reservoir,
    ReservoirHyperparameters,
    HybridReservoirComputingModel,
)


class MultiOutputMeanRegressor:
    def __init__(self, n_outputs: int):
        self.n_outputs = n_outputs

    def predict(self, input):
        # returns array of shape (1, n_outputs), with each element
        # the mean of the input vector elements
        return np.full(self.n_outputs, np.mean(input)).reshape(1, -1)


def _sparse_allclose(A, B, atol=1e-8):
    # https://stackoverflow.com/a/47771340
    r1, c1, v1 = sparse.find(A)
    r2, c2, v2 = sparse.find(B)
    index_match = np.array_equal(r1, r2) & np.array_equal(c1, c2)
    if index_match == 0:
        return False
    else:
        return np.allclose(v1, v2, atol=atol)


def test_dump_load_optional_attrs(tmpdir):
    input_size = 10
    hyperparameters = ReservoirHyperparameters(
        state_size=100,
        adjacency_matrix_sparsity=0.0,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=input_size)
    readout = ReservoirComputingReadout(
        coefficients=sparse.coo_matrix(np.random.rand(input_size, 100)),
        intercepts=np.random.rand(input_size),
    )
    rank_divider = RankDivider([2, 2], ["x", "y", "z"], [2, 2, 2], 2)
    predictor = ReservoirComputingModel(
        input_variables=["a", "b"],
        output_variables=["a", "b"],
        reservoir=reservoir,
        readout=readout,
        square_half_hidden_state=False,
        rank_divider=rank_divider,
    )
    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)
    loaded_predictor = ReservoirComputingModel.load(output_path)

    assert loaded_predictor.rank_divider is not None


def test_dump_load_preserves_matrices(tmpdir):
    input_size = 10
    state_size = 150
    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.0,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=input_size)
    readout = ReservoirComputingReadout(
        coefficients=sparse.coo_matrix(np.random.rand(input_size, state_size)),
        intercepts=np.random.rand(input_size),
    )
    predictor = ReservoirComputingModel(
        input_variables=["a", "b"],
        output_variables=["a", "b"],
        reservoir=reservoir,
        readout=readout,
        square_half_hidden_state=False,
    )
    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)

    loaded_predictor = ReservoirComputingModel.load(output_path)
    assert _sparse_allclose(loaded_predictor.reservoir.W_in, predictor.reservoir.W_in,)
    assert _sparse_allclose(
        loaded_predictor.reservoir.W_res, predictor.reservoir.W_res,
    )
    assert _sparse_allclose(
        loaded_predictor.readout.coefficients, predictor.readout.coefficients,
    )
    np.testing.assert_array_almost_equal(
        loaded_predictor.readout.intercepts, predictor.readout.intercepts
    )


def test_prediction_shape():
    input_size = 15
    state_size = 1000
    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.9,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=input_size)
    reservoir.reset_state(input_shape=(input_size,))
    readout = ReservoirComputingReadout(
        coefficients=np.random.rand(state_size, input_size),
        intercepts=np.random.rand(input_size),
    )
    predictor = ReservoirComputingModel(
        input_variables=["a", "b"],
        output_variables=["a", "b"],
        reservoir=reservoir,
        readout=readout,
    )
    # ReservoirComputingModel.predict reshapes the prediction to remove
    # the first dim of length 1 (sklearn regressors predict 2D arrays)
    assert predictor.predict().shape == (input_size,)


def test_ReservoirComputingModel_state_increment():
    input_size = 2
    state_size = 3
    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.0,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=input_size)
    reservoir.W_in = sparse.coo_matrix(np.ones(reservoir.W_in.shape))
    reservoir.W_res = sparse.coo_matrix(np.ones(reservoir.W_res.shape))

    readout = MultiOutputMeanRegressor(n_outputs=input_size)
    predictor = ReservoirComputingModel(
        input_variables=["a", "b"],
        output_variables=["a", "b"],
        reservoir=reservoir,
        readout=readout,
    )

    input = np.array([0.5, 0.5])
    predictor.reset_state()
    predictor.reservoir.increment_state(input)
    state_before_prediction = predictor.reservoir.state
    prediction = predictor.predict()
    predictor.increment_state(prediction)
    np.testing.assert_array_almost_equal(prediction, np.tanh(np.array([1.0, 1.0])))
    assert not np.allclose(state_before_prediction, predictor.reservoir.state)


def test_prediction_after_load(tmpdir):
    input_size = 15
    state_size = 1000
    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.9,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=input_size)

    readout = ReservoirComputingReadout(
        coefficients=np.random.rand(state_size, input_size),
        intercepts=np.random.rand(input_size),
    )
    predictor = ReservoirComputingModel(
        input_variables=["a", "b"],
        output_variables=["a", "b"],
        reservoir=reservoir,
        readout=readout,
    )
    predictor.reset_state()

    ts_sync = [np.ones(input_size) for i in range(20)]
    predictor.synchronize(ts_sync)
    for i in range(10):
        prediction0 = predictor.predict()

    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)
    loaded_predictor = ReservoirComputingModel.load(output_path)
    loaded_predictor.reset_state()

    loaded_predictor.synchronize(ts_sync)
    for i in range(10):
        prediction1 = loaded_predictor.predict()

    np.testing.assert_array_almost_equal(prediction0, prediction1)


def test_HybridReservoirComputingModel_dump_load(tmpdir):
    input_size = 15
    hybrid_input_size = 4
    state_size = 1000
    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.9,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=input_size)

    readout = ReservoirComputingReadout(
        coefficients=np.random.rand(state_size + hybrid_input_size, input_size),
        intercepts=np.random.rand(input_size),
    )
    hybrid_predictor = HybridReservoirComputingModel(
        input_variables=["a", "b"],
        hybrid_variables=["c"],
        output_variables=["a", "b"],
        reservoir=reservoir,
        readout=readout,
    )
    hybrid_predictor.reset_state()
    ts_sync = [np.ones(input_size) for i in range(20)]

    hybrid_predictor.synchronize(ts_sync)

    n_predict = 7
    hybrid_input = np.random.rand(n_predict, hybrid_input_size)
    for i in range(n_predict):
        prediction0 = hybrid_predictor.predict(hybrid_input[i])

    output_path = f"{str(tmpdir)}/predictor"
    hybrid_predictor.dump(output_path)
    loaded_hybrid_predictor = HybridReservoirComputingModel.load(output_path)
    loaded_hybrid_predictor.reset_state()

    loaded_hybrid_predictor.synchronize(ts_sync)
    for i in range(n_predict):
        prediction1 = loaded_hybrid_predictor.predict(hybrid_input[i])

    np.testing.assert_array_almost_equal(prediction0, prediction1)


def test_HybridReservoirComputingModel_concat_readout_inputs():
    # TODO: Complex test, will be simplified in future when the model is refactored
    input_size = 4
    state_size = 3
    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.9,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    rank_divider = RankDivider([2, 2], ["x", "y", "z"], [2, 2, 1], 0)

    reservoir = Reservoir(hyperparameters, input_size=rank_divider.n_subdomain_features)

    readout = ReservoirComputingReadout(
        coefficients=np.random.rand(
            state_size + rank_divider.n_subdomain_features, input_size
        ),
        intercepts=np.random.rand(input_size),
    )
    hybrid_predictor = HybridReservoirComputingModel(
        input_variables=["a", "b"],
        hybrid_variables=["c"],
        output_variables=["a", "b"],
        reservoir=reservoir,
        readout=readout,
        rank_divider=rank_divider,
    )
    hybrid_predictor.reset_state()

    # hidden state of each subdomain is constant array of its index
    hybrid_predictor.reservoir_model.reservoir.state = np.array(
        [np.arange(rank_divider.n_subdomains) for zfeature in range(state_size)]
    )

    # partitioner indexing goes (0,0) -> 0, (1,0)-> 1, etc.
    hybrid_inputs = np.array([[0, -2], [-1, -3]])

    flattened_readout_input = hybrid_predictor._concatenate_readout_inputs(
        hybrid_predictor.reservoir_model.reservoir.state, hybrid_inputs
    )
    np.testing.assert_array_equal(
        flattened_readout_input,
        np.array([0, 0, 0, 0, 1, 1, 1, -1, 2, 2, 2, -2, 3, 3, 3, -3]),
    )
