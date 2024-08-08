from abc import ABCMeta


class PreInitMeta(type):
    """
    Metaclass that ensures all base classes' __init__ methods are called,
    even if the child class does not explicitly call them.
    """

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)

        # Store called __init__ methods to avoid duplicate calls
        called_inits = set()

        def call_base_inits(base):
            if base is object:
                return
            for parent in base.__bases__:
                call_base_inits(parent)
            if (
                base.__init__ is not object.__init__
                and base.__init__ not in called_inits
            ):
                base.__init__(instance, *args, **kwargs)
                called_inits.add(base.__init__)

        for base in cls.__bases__:
            call_base_inits(base)

        return instance


class PreInitABCMeta(PreInitMeta, ABCMeta):
    pass
