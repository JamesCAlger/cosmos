"""Component descriptor system for self-describing components"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from loguru import logger


class ParamType(Enum):
    """Parameter types for component configuration"""
    INT = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOL = "boolean"
    CHOICE = "choice"
    LIST = "list"


@dataclass
class ParamSpec:
    """Specification for a component parameter"""
    type: ParamType
    default: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    description: Optional[str] = None
    tunable: bool = True  # Can be optimized
    required: bool = False

    def __post_init__(self):
        """Validate parameter specification"""
        # Validate choice parameters
        if self.type == ParamType.CHOICE and not self.choices:
            raise ValueError("CHOICE type requires 'choices' to be specified")

        # Validate numeric parameters
        if self.type in [ParamType.INT, ParamType.FLOAT]:
            if self.min_value is not None and self.max_value is not None:
                if self.min_value > self.max_value:
                    raise ValueError(f"min_value ({self.min_value}) cannot be greater than max_value ({self.max_value})")

        # Validate default value
        if self.default is not None:
            if not self.validate_value(self.default):
                raise ValueError(f"Default value {self.default} is invalid for parameter spec")

    def validate_value(self, value: Any) -> bool:
        """Validate a value against this parameter specification"""
        if value is None and not self.required:
            return True

        if value is None and self.required:
            return False

        # Type checking
        if self.type == ParamType.INT:
            if not isinstance(value, int):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

        elif self.type == ParamType.FLOAT:
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

        elif self.type == ParamType.STRING:
            if not isinstance(value, str):
                return False

        elif self.type == ParamType.BOOL:
            if not isinstance(value, bool):
                return False

        elif self.type == ParamType.CHOICE:
            if value not in self.choices:
                return False

        elif self.type == ParamType.LIST:
            if not isinstance(value, list):
                return False

        return True

    def to_search_space(self) -> Dict[str, Any]:
        """Convert to search space representation for optimization"""
        if not self.tunable:
            return None

        if self.type == ParamType.CHOICE:
            return {"type": "categorical", "choices": self.choices}
        elif self.type == ParamType.INT:
            return {
                "type": "integer",
                "min": self.min_value or 0,
                "max": self.max_value or 100
            }
        elif self.type == ParamType.FLOAT:
            return {
                "type": "continuous",
                "min": self.min_value or 0.0,
                "max": self.max_value or 1.0
            }
        elif self.type == ParamType.BOOL:
            return {"type": "categorical", "choices": [True, False]}

        return None


@dataclass
class ComponentDescriptor:
    """Complete description of a component's capabilities"""
    # Component identity
    name: str
    type: str  # retriever, generator, chunker, etc.
    version: str = "1.0.0"

    # Parameters
    parameters: Dict[str, ParamSpec] = field(default_factory=dict)

    # Pipeline integration
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    # Optimization hints
    tunable_params: List[str] = field(default_factory=list)

    # Dependencies and constraints
    requires: Optional[List[str]] = None
    parallel_with: Optional[List[str]] = None
    mutually_exclusive: Optional[List[str]] = None

    # Performance hints
    estimated_cost: float = 0.0  # Per-call cost estimate in dollars
    estimated_latency: float = 0.0  # Seconds
    batch_capable: bool = False

    def __post_init__(self):
        """Validate descriptor consistency"""
        # Ensure tunable params exist in parameters
        for param_name in self.tunable_params:
            if param_name not in self.parameters:
                raise ValueError(f"Tunable parameter '{param_name}' not found in parameters")

        # Auto-populate tunable_params if empty but parameters are tunable
        if not self.tunable_params:
            self.tunable_params = [
                name for name, spec in self.parameters.items()
                if spec.tunable
            ]

        logger.debug(f"ComponentDescriptor created for {self.name} ({self.type})")

    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate configuration against descriptor

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required parameters
        for param_name, param_spec in self.parameters.items():
            if param_spec.required and param_name not in config:
                errors.append(f"Required parameter '{param_name}' is missing")
                continue

            # Validate parameter value if present
            if param_name in config:
                value = config[param_name]
                if not param_spec.validate_value(value):
                    errors.append(f"Invalid value for '{param_name}': {value}")

        # Check for unknown parameters
        for param_name in config:
            if param_name not in self.parameters:
                errors.append(f"Unknown parameter '{param_name}'")

        return len(errors) == 0, errors

    def get_search_space(self) -> Dict[str, Any]:
        """Generate search space for optimization"""
        search_space = {}

        for param_name in self.tunable_params:
            param_spec = self.parameters[param_name]
            space_def = param_spec.to_search_space()
            if space_def:
                search_space[param_name] = space_def

        return search_space

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration from parameter specs"""
        config = {}
        for param_name, param_spec in self.parameters.items():
            if param_spec.default is not None:
                config[param_name] = param_spec.default
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptor to dictionary representation"""
        return {
            "name": self.name,
            "type": self.type,
            "version": self.version,
            "parameters": {
                name: {
                    "type": spec.type.value,
                    "default": spec.default,
                    "tunable": spec.tunable,
                    "required": spec.required,
                    "description": spec.description
                }
                for name, spec in self.parameters.items()
            },
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tunable_params": self.tunable_params,
            "requires": self.requires,
            "parallel_with": self.parallel_with,
            "mutually_exclusive": self.mutually_exclusive,
            "estimated_cost": self.estimated_cost,
            "estimated_latency": self.estimated_latency,
            "batch_capable": self.batch_capable
        }


class SelfDescribingComponent:
    """Mixin for self-describing components"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        """Return component descriptor - must be implemented by subclasses"""
        raise NotImplementedError(f"{cls.__name__} must implement describe() method")

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against descriptor"""
        descriptor = self.__class__.describe()
        is_valid, errors = descriptor.validate_config(config)

        if not is_valid:
            logger.error(f"Configuration validation failed for {self.__class__.__name__}:")
            for error in errors:
                logger.error(f"  - {error}")

        return is_valid

    def get_config_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge provided config with defaults from descriptor"""
        descriptor = self.__class__.describe()
        default_config = descriptor.get_default_config()

        # Merge configs (provided config overrides defaults)
        merged = default_config.copy()
        merged.update(config)

        return merged