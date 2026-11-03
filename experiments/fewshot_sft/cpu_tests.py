"""Run CPU tests and record actual pass/skip/failure evidence."""
import unittest
import sys
import platform
from .common import OUT, write


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('experiments.fewshot_sft.test_pipeline'))
    write(OUT/'cpu_tests.json', dict(tests=result.testsRun, executable=sys.executable,
        python=platform.python_version(),
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        skipped=[(str(t), reason) for t,reason in result.skipped],
        failures=[(str(t), reason) for t,reason in result.failures],
        errors=[(str(t), reason) for t,reason in result.errors],
        successful=result.wasSuccessful(), scope='CPU fixtures and source contracts; not a real RLDS/GPU integration test'), replace=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
