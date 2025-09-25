"""Tests for component descriptor system"""

import pytest
from typing import Dict, Any

from autorag.components.descriptors import (
    ParamType,
    ParamSpec,
    ComponentDescriptor,
    SelfDescribingComponent
)


class TestParamSpec:
    """Test parameter specifications"""

    def test_integer_param_spec(self):
        """Test integer parameter specification"""
        spec = ParamSpec(
            type=ParamType.INT,
            default=10,
            min_value=1,
            max_value=100,
            description="Test integer parameter"
        )

        # Valid values
        assert spec.validate_value(10) is True
        assert spec.validate_value(1) is True
        assert spec.validate_value(100) is True

        # Invalid values
        assert spec.validate_value(0) is False
        assert spec.validate_value(101) is False
        assert spec.validate_value(10.5) is False
        assert spec.validate_value("10") is False

    def test_float_param_spec(self):
        """Test float parameter specification"""
        spec = ParamSpec(
            type=ParamType.FLOAT,
            default=0.5,
            min_value=0.0,
            max_value=1.0
        )

        # Valid values
        assert spec.validate_value(0.5) is True
        assert spec.validate_value(0.0) is True
        assert spec.validate_value(1.0) is True
        assert spec.validate_value(1) is True  # int should work for float

        # Invalid values
        assert spec.validate_value(-0.1) is False
        assert spec.validate_value(1.1) is False
        assert spec.validate_value("0.5") is False

    def test_choice_param_spec(self):
        """Test choice parameter specification"""
        spec = ParamSpec(
            type=ParamType.CHOICE,
            default="option1",
            choices=["option1", "option2", "option3"]
        )

        # Valid values
        assert spec.validate_value("option1") is True
        assert spec.validate_value("option2") is True

        # Invalid values
        assert spec.validate_value("option4") is False
        assert spec.validate_value(1) is False

    def test_choice_param_requires_choices(self):
        """Test that choice parameters require choices to be specified"""
        with pytest.raises(ValueError, match="CHOICE type requires 'choices'"):
            ParamSpec(
                type=ParamType.CHOICE,
                default="option1"
                # Missing choices
            )

    def test_boolean_param_spec(self):
        """Test boolean parameter specification"""
        spec = ParamSpec(
            type=ParamType.BOOL,
            default=True
        )

        assert spec.validate_value(True) is True
        assert spec.validate_value(False) is True
        assert spec.validate_value(1) is False
        assert spec.validate_value("true") is False

    def test_required_param(self):
        """Test required parameter validation"""
        spec = ParamSpec(
            type=ParamType.STRING,
            default=None,
            required=True
        )

        assert spec.validate_value("value") is True
        assert spec.validate_value(None) is False

    def test_optional_param(self):
        """Test optional parameter validation"""
        spec = ParamSpec(
            type=ParamType.STRING,
            default=None,
            required=False
        )

        assert spec.validate_value("value") is True
        assert spec.validate_value(None) is True

    def test_invalid_numeric_range(self):
        """Test that min > max raises error"""
        with pytest.raises(ValueError, match="min_value .* cannot be greater than max_value"):
            ParamSpec(
                type=ParamType.INT,
                default=10,
                min_value=100,
                max_value=10
            )

    def test_invalid_default_value(self):
        """Test that invalid default value raises error"""
        with pytest.raises(ValueError, match="Default value .* is invalid"):
            ParamSpec(
                type=ParamType.INT,
                default=200,  # Outside range
                min_value=1,
                max_value=100
            )

    def test_to_search_space(self):
        """Test conversion to search space representation"""
        # Integer param
        int_spec = ParamSpec(
            type=ParamType.INT,
            default=10,
            min_value=1,
            max_value=100,
            tunable=True
        )
        space = int_spec.to_search_space()
        assert space["type"] == "integer"
        assert space["min"] == 1
        assert space["max"] == 100

        # Choice param
        choice_spec = ParamSpec(
            type=ParamType.CHOICE,
            default="a",
            choices=["a", "b", "c"],
            tunable=True
        )
        space = choice_spec.to_search_space()
        assert space["type"] == "categorical"
        assert space["choices"] == ["a", "b", "c"]

        # Non-tunable param
        fixed_spec = ParamSpec(
            type=ParamType.STRING,
            default="fixed",
            tunable=False
        )
        assert fixed_spec.to_search_space() is None


class TestComponentDescriptor:
    """Test component descriptors"""

    def test_basic_descriptor(self):
        """Test basic descriptor creation"""
        descriptor = ComponentDescriptor(
            name="TestComponent",
            type="processor",
            version="1.0.0",
            parameters={
                "param1": ParamSpec(type=ParamType.INT, default=10),
                "param2": ParamSpec(type=ParamType.STRING, default="test")
            },
            inputs=["input1"],
            outputs=["output1"]
        )

        assert descriptor.name == "TestComponent"
        assert descriptor.type == "processor"
        assert len(descriptor.parameters) == 2

    def test_auto_populate_tunable_params(self):
        """Test automatic population of tunable parameters"""
        descriptor = ComponentDescriptor(
            name="TestComponent",
            type="processor",
            parameters={
                "tunable_param": ParamSpec(
                    type=ParamType.INT,
                    default=10,
                    tunable=True
                ),
                "fixed_param": ParamSpec(
                    type=ParamType.STRING,
                    default="fixed",
                    tunable=False
                )
            }
        )

        # Should auto-populate tunable_params
        assert "tunable_param" in descriptor.tunable_params
        assert "fixed_param" not in descriptor.tunable_params

    def test_invalid_tunable_param_reference(self):
        """Test that referencing non-existent parameter in tunable_params raises error"""
        with pytest.raises(ValueError, match="Tunable parameter .* not found"):
            ComponentDescriptor(
                name="TestComponent",
                type="processor",
                parameters={
                    "param1": ParamSpec(type=ParamType.INT, default=10)
                },
                tunable_params=["nonexistent_param"]
            )

    def test_validate_config(self):
        """Test configuration validation"""
        descriptor = ComponentDescriptor(
            name="TestComponent",
            type="processor",
            parameters={
                "required_param": ParamSpec(
                    type=ParamType.STRING,
                    default=None,
                    required=True
                ),
                "optional_param": ParamSpec(
                    type=ParamType.INT,
                    default=10,
                    min_value=0,
                    max_value=100,
                    required=False
                )
            }
        )

        # Valid config
        is_valid, errors = descriptor.validate_config({
            "required_param": "value",
            "optional_param": 50
        })
        assert is_valid is True
        assert len(errors) == 0

        # Missing required parameter
        is_valid, errors = descriptor.validate_config({
            "optional_param": 50
        })
        assert is_valid is False
        assert "Required parameter 'required_param' is missing" in errors

        # Invalid parameter value
        is_valid, errors = descriptor.validate_config({
            "required_param": "value",
            "optional_param": 200  # Outside range
        })
        assert is_valid is False
        assert any("Invalid value for 'optional_param'" in e for e in errors)

        # Unknown parameter
        is_valid, errors = descriptor.validate_config({
            "required_param": "value",
            "unknown_param": "value"
        })
        assert is_valid is False
        assert "Unknown parameter 'unknown_param'" in errors

    def test_get_search_space(self):
        """Test search space generation"""
        descriptor = ComponentDescriptor(
            name="TestComponent",
            type="processor",
            parameters={
                "int_param": ParamSpec(
                    type=ParamType.INT,
                    default=10,
                    min_value=1,
                    max_value=100,
                    tunable=True
                ),
                "choice_param": ParamSpec(
                    type=ParamType.CHOICE,
                    default="a",
                    choices=["a", "b", "c"],
                    tunable=True
                ),
                "fixed_param": ParamSpec(
                    type=ParamType.STRING,
                    default="fixed",
                    tunable=False
                )
            },
            tunable_params=["int_param", "choice_param"]
        )

        search_space = descriptor.get_search_space()

        assert "int_param" in search_space
        assert search_space["int_param"]["type"] == "integer"
        assert search_space["int_param"]["min"] == 1
        assert search_space["int_param"]["max"] == 100

        assert "choice_param" in search_space
        assert search_space["choice_param"]["type"] == "categorical"
        assert search_space["choice_param"]["choices"] == ["a", "b", "c"]

        assert "fixed_param" not in search_space

    def test_get_default_config(self):
        """Test default configuration extraction"""
        descriptor = ComponentDescriptor(
            name="TestComponent",
            type="processor",
            parameters={
                "param1": ParamSpec(type=ParamType.INT, default=10),
                "param2": ParamSpec(type=ParamType.STRING, default="test"),
                "param3": ParamSpec(type=ParamType.BOOL, default=None)
            }
        )

        config = descriptor.get_default_config()

        assert config["param1"] == 10
        assert config["param2"] == "test"
        assert "param3" not in config  # None defaults are not included

    def test_descriptor_to_dict(self):
        """Test conversion to dictionary"""
        descriptor = ComponentDescriptor(
            name="TestComponent",
            type="processor",
            version="2.0.0",
            parameters={
                "param1": ParamSpec(
                    type=ParamType.INT,
                    default=10,
                    description="Test parameter"
                )
            },
            inputs=["input1"],
            outputs=["output1"],
            estimated_cost=0.01,
            estimated_latency=0.5
        )

        d = descriptor.to_dict()

        assert d["name"] == "TestComponent"
        assert d["version"] == "2.0.0"
        assert d["estimated_cost"] == 0.01
        assert d["estimated_latency"] == 0.5
        assert "param1" in d["parameters"]
        assert d["parameters"]["param1"]["type"] == "integer"


class TestSelfDescribingComponent:
    """Test self-describing component mixin"""

    def test_validate_config_with_mixin(self):
        """Test configuration validation using mixin"""

        class TestComponent(SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="TestComponent",
                    type="test",
                    parameters={
                        "param": ParamSpec(
                            type=ParamType.INT,
                            default=10,
                            min_value=1,
                            max_value=100
                        )
                    }
                )

            def __init__(self, config: Dict[str, Any]):
                self.config = config

        component = TestComponent({"param": 50})

        # Valid config
        assert component.validate_config({"param": 50}) is True

        # Invalid config
        assert component.validate_config({"param": 200}) is False

    def test_get_config_with_defaults(self):
        """Test merging config with defaults"""

        class TestComponent(SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="TestComponent",
                    type="test",
                    parameters={
                        "param1": ParamSpec(type=ParamType.INT, default=10),
                        "param2": ParamSpec(type=ParamType.STRING, default="default"),
                        "param3": ParamSpec(type=ParamType.BOOL, default=True)
                    }
                )

        component = TestComponent()

        # Partial config
        merged = component.get_config_with_defaults({"param1": 20})
        assert merged["param1"] == 20  # Overridden
        assert merged["param2"] == "default"  # Default
        assert merged["param3"] is True  # Default

        # Empty config
        merged = component.get_config_with_defaults({})
        assert merged["param1"] == 10
        assert merged["param2"] == "default"
        assert merged["param3"] is True

    def test_not_implemented_describe(self):
        """Test that describe() must be implemented"""

        class BadComponent(SelfDescribingComponent):
            pass

        with pytest.raises(NotImplementedError):
            BadComponent.describe()