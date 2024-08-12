from abc import ABCMeta


class PreInitMeta(type):
    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls)

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

        # Call parent initializers first
        for base in cls.__bases__:
            call_base_inits(base)

        # Now call the current class's __init__
        if cls.__init__ is not object.__init__:
            cls.__init__(instance, *args, **kwargs)

        return instance


class PreInitABCMeta(PreInitMeta, ABCMeta):
    pass
