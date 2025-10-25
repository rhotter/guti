"""
Utility functions for notebook-style Python files.

This module provides functions to enable Jupyter notebook features
in regular Python files without causing linter errors.
"""

import matplotlib.pyplot as plt


def enable_autoreload():
    """
    Enable IPython autoreload extension if running in IPython/Jupyter.

    This is equivalent to:
    %load_ext autoreload
    %autoreload 2

    But works in regular Python files without causing linter errors.
    """
    try:
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
    except (NameError, AttributeError):
        # get_ipython() is not defined, we're not in IPython
        pass


def white_background():
    """
    Set the background color of the figure to white.
    """
    plt.rcParams["figure.facecolor"] = "white"  # Figure background
    plt.rcParams["axes.facecolor"] = "white"  # Axes background
    plt.rcParams["savefig.facecolor"] = "white"  # Saved figure background
    plt.rcParams["savefig.transparent"] = False  # Ensure not transparent


def notebook_setup():
    """
    Common setup for notebook-style files.

    Enables autoreload and any other common notebook configurations.
    """
    enable_autoreload()
    white_background()


# Convenience function that can be called directly
setup = notebook_setup