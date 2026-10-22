"""No GPU or simulator: verify diagnostic reset behavior and exception boundary."""
import unittest
from diagnose_goal_stall import DiagnosticAbort, visible_reset


class RandomizationError(Exception):
    pass


class ResetTests(unittest.TestCase):
    def test_success_preserves_result(self):
        value = object()
        self.assertIs(visible_reset(lambda: value, RandomizationError, lambda *a, **k: None, 3), value)

    def test_randomization_retry_preserves_order(self):
        calls = []
        def reset():
            calls.append(1)
            if len(calls) == 1:
                raise RandomizationError('placement failed')
            return 'observation'
        events = []
        result = visible_reset(reset, RandomizationError, lambda event, **kw: events.append(event), 3)
        self.assertEqual(result, 'observation')
        self.assertEqual(events, ['reset_attempt', 'reset_randomization_error', 'reset_attempt'])

    def test_retry_limit_aborts(self):
        calls = []
        def reset():
            calls.append(1)
            raise RandomizationError('placement failed')
        with self.assertRaises(DiagnosticAbort):
            visible_reset(reset, RandomizationError, lambda *a, **k: None, 3)
        self.assertEqual(len(calls), 3)

    def test_unexpected_exception_is_visible_and_not_policy_failure(self):
        events = []
        def reset():
            raise ValueError('underlying simulator error')
        try:
            visible_reset(reset, RandomizationError, lambda event, **kw: events.append((event, kw)), 3)
        except Exception:
            self.fail('Would be swallowed by official episode except Exception')
        except DiagnosticAbort as exc:
            self.assertIsInstance(exc.__cause__, ValueError)
        else:
            self.fail('Expected diagnostic abort')
        self.assertIn('underlying simulator error', events[-1][1]['traceback'])

    def test_keyboard_interrupt_is_not_swallowed(self):
        def reset():
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            visible_reset(reset, RandomizationError, lambda *a, **k: None, 3)


if __name__ == '__main__':
    unittest.main()
