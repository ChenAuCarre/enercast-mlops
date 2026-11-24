from datetime import datetime

from pydantic import BaseModel, Field, create_model, field_validator

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _bounds(lo=None, hi=None):
    return {k: v for k, v in (("ge", lo), ("le", hi)) if v is not None}


_FIELDS = {"lights": _bounds(0)}
_FIELDS.update({f"T{i}": _bounds(-20, 60) for i in range(1, 10)})
_FIELDS.update({f"RH_{i}": _bounds(0, 100) for i in range(1, 10)})
_FIELDS.update(
    To=_bounds(-40, 60),
    Pressure=_bounds(650, 850),
    RH_out=_bounds(0, 100),
    Windspeed=_bounds(0),
    Visibility=_bounds(0),
    Tdewpoint=_bounds(-50, 50),
)


class _PredictBase(BaseModel):
    date: str

    @field_validator("date")
    @classmethod
    def _check_date(cls, v):
        datetime.strptime(v, DATE_FORMAT)  # ValueError -> 422
        return v


# Bounds follow docs/API.md; all fields required.
PredictRequest = create_model(
    "PredictRequest",
    __base__=_PredictBase,
    **{name: (float, Field(..., **b)) for name, b in _FIELDS.items()},
)


class PredictResponse(BaseModel):
    predicted_appliances_wh: float
    model_name: str
    model_alias: str
    model_version: str


class RefreshResponse(BaseModel):
    status: str
    previous_version: str | None
    current_version: str
