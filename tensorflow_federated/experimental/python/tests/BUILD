load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_test(
    name = "jax_training_test",
    srcs = ["jax_training_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)
