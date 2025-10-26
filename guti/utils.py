def enable_autoreload():
    """
    Enable automatic reloading of modules in Jupyter notebooks.

    This function should be called at the beginning of interactive Jupyter notebooks
    to automatically reload any modified code from the repository without restarting
    the kernel.

    Uses IPython's autoreload magic:
    - %load_ext autoreload: Loads the autoreload extension
    - %autoreload 2: Reloads all modules before executing code

    Example
    -------
    >>> from utils import enable_autoreload
    >>> enable_autoreload()
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()

        if ipython is not None:
            # Use run_line_magic instead of deprecated magic() method
            ipython.run_line_magic('load_ext', 'autoreload')
            ipython.run_line_magic('autoreload', '2')
            print("Autoreload enabled: modules will be automatically reloaded when modified")
        else:
            print("Warning: Not running in IPython/Jupyter environment, autoreload not enabled")
    except ImportError:
        print("Warning: IPython not available, autoreload not enabled")
    except Exception as e:
        print(f"Warning: Could not enable autoreload: {e}")