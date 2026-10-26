"""Signing and verification boundaries shared by protocol messages."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Signer(Protocol):
    @property
    def validator_hotkey(self) -> str: ...

    def sign(self, message: bytes) -> bytes: ...


@runtime_checkable
class Verifier(Protocol):
    def verify(self, validator_hotkey: str, message: bytes, signature: bytes) -> bool: ...
