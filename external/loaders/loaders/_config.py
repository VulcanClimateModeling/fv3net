import abc
from typing import Dict, TypeVar, Callable
from loaders.typing import (
    Mapper,
    Batches,
)
import dataclasses
import dacite


RT = TypeVar("RT")


class FunctionRegister(Dict[str, Callable[..., RT]]):
    def register(self, func: Callable[..., RT]) -> Callable[..., RT]:
        self[func.__name__] = func
        return func

    def __repr__(self):
        return str(sorted(list(self.keys())))


mapper_functions: FunctionRegister[Mapper] = FunctionRegister()
batches_functions: FunctionRegister[Batches] = FunctionRegister()
batches_from_mapper_functions: FunctionRegister[Batches] = FunctionRegister()


@dataclasses.dataclass
class MapperConfig:
    """Configuration for the use of mapper loading functions.

    Attributes:
        data_path: location of training data to be loaded by mapper function
        mapper_function: name of function to use for loading batched data,
            can take any value in the keys of `loaders.mapper_functions`
        mapper_kwargs: keyword arguments to pass to mapper function
    """

    data_path: str
    mapper_function: str
    mapper_kwargs: dict

    def load_mapper(self) -> Mapper:
        """
        Returns:
            Sequence of mappers according to configuration
        """
        mapping_func = mapper_functions[self.mapper_function]
        return mapping_func(self.data_path, **self.mapper_kwargs)

    def __post_init__(self):
        if self.mapper_function not in mapper_functions:
            raise ValueError(
                f"Invalid mapper function {self.mapper_function}, "
                f"must be one of {list(mapper_functions.keys())}"
            )


class BatchesLoader(abc.ABC):
    @abc.abstractmethod
    def load_batches(self, variables) -> Batches:
        """
        Args:
            config: data configuration

        Returns:
            Sequence of datasets according to configuration
        """
        pass

    @classmethod
    def from_dict(cls, kwargs) -> "BatchesLoader":
        try:
            return dacite.from_dict(data_class=cls, data=kwargs)
        except (TypeError, AttributeError):
            pass
        for subclass in cls.__subclasses__():
            try:
                return dacite.from_dict(data_class=subclass, data=kwargs)
            except (TypeError, AttributeError, dacite.exceptions.MissingValueError):
                pass
        raise ValueError("invalid BatchesLoader dictionary")


@dataclasses.dataclass
class BatchesFromMapperConfig(BatchesLoader):
    """Configuration for the use of batch loading functions using mappers as input.

    Attributes:
        mapper_config: configuration to retriev einput mapper
        batches_function: name of function to use to convert mapper to batches,
            can take any value in the keys of `loaders.batches_from_mapper_functions`
        batches_kwargs: keyword arguments to pass to batches function
    """

    mapper_config: MapperConfig
    batches_function: str
    batches_kwargs: dict

    def load_mapper(self) -> Mapper:
        return self.mapper_config.load_mapper()

    def load_batches(self, variables) -> Batches:
        """
        Args:
            variables: names of variables to include in dataset

        Returns:
            Sequence of datasets according to configuration
        """
        mapper = self.mapper_config.load_mapper()
        batches_function = batches_from_mapper_functions[self.batches_function]
        return batches_function(mapper, list(variables), **self.batches_kwargs,)

    def __post_init__(self):
        if self.batches_function not in batches_from_mapper_functions:
            raise ValueError(
                f"Invalid batches function {self.batches_function}, "
                f"must be one of {list(batches_from_mapper_functions.keys())}"
            )


@dataclasses.dataclass
class BatchesConfig(BatchesLoader):
    """Configuration for the use of batch loading functions.

    Attributes:
        data_path: location of training data to be loaded by batch function
        batches_function: name of function to use for loading batched data,
            can take any value in the keys of `loaders.batches_functions`
        batches_kwargs: keyword arguments to pass to batches function
    """

    data_path: str
    batches_function: str
    batches_kwargs: dict

    def load_batches(self, variables) -> Batches:
        """
        Args:
            variables: names of variables to include in dataset

        Returns:
            Sequence of datasets according to configuration
        """
        batches_function = batches_functions[self.batches_function]
        return batches_function(self.data_path, list(variables), **self.batches_kwargs,)

    def __post_init__(self):
        if self.batches_function not in batches_functions:
            raise ValueError(
                f"Invalid batches function {self.batches_function}, "
                f"must be one of {list(batches_functions.keys())}"
            )
