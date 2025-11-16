import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings


warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered"
)


def Func1(x):
    return 0.1 * x + 0.5 * np.cos(0.1 * x) - 0.3


def Func2(x):
    return 9.8 * x - np.exp(0.6 * x)


def IterProcess1(x):
    return (0.3 - 0.5 * np.cos(0.1 * x)) / (0.1)


def IterProcess2(x):
    return (np.exp(0.6 * x)) / (9.8)


def ClarifyRoots(func, search_range=(-10, 10), step=0.01):
    x = np.arange(search_range[0], search_range[1], step)
    y = np.array([func(xi) for xi in x])

    root_intervals = []

    for i in range(len(y) - 1):
        if np.sign(y[i]) != np.sign(y[i + 1]):
            root_intervals.append((x[i], x[i + 1]))

    return root_intervals


def DichotomyMethod(func, rootIntervals, eps, max_iter=1000):
    roots = []

    for interval in rootIntervals:
        a, b = interval
        iters = 0
        fa = func(a)
        fb = func(b)

        while (b - a) > eps and iters < max_iter:
            c = (a + b) / 2
            fc = func(c)

            if fc == 0:
                break
            elif fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

            iters += 1

        roots.append((a + b) / 2)

    return roots


def SimpleIteration(iterProc, rootIntervals, eps, max_iter=1000):
    roots = []

    for interval in rootIntervals:
        a, b = interval
        iters = 0
        divergence_flag = False

        x0 = (a + b) / 2

        try:

            xi = iterProc(x0)

            while abs(xi - x0) > eps and iters < max_iter:
                if not np.isfinite(xi):
                    divergence_flag = True
                    break

                xn = iterProc(xi)
                x0 = xi
                xi = xn
                iters += 1

            if divergence_flag:
                roots.append(f"Расходится (Inf) на ({a:.4f}, {b:.4f})")
            else:
                roots.append(xn)

        except Exception as e:
            roots.append(f"Ошибка: {str(e)}")

    return roots


def NumericDeriv1(func, x, h=1e-6):
    return (func(x + h) - func(x - h)) / (2 * h)


def NewtonRoots(func, rootIntervals, eps, max_iter=1000):
    roots = []

    for interval in rootIntervals:
        a, b = interval
        iters = 0
        divergence_flag = False

        x0 = (a + b) / 2

        try:

            xi = x0 - (func(x0) / NumericDeriv1(func, x0))

            while abs(xi - x0) > eps and iters < max_iter:
                if not np.isfinite(xi):
                    divergence_flag = True
                    break

                xn = xi - (func(xi) / NumericDeriv1(func, xi))
                x0 = xi
                xi = xn
                iters += 1

            if divergence_flag:
                roots.append(f"Расходится (Inf) на ({a:.4f}, {b:.4f})")
            else:
                roots.append(xn)

        except Exception as e:
            roots.append(f"Ошибка: {str(e)}")

    return roots


def SciPyRoots(func, rootIntervals):
    roots = []
    for a, b in rootIntervals:
        x0 = (a + b) / 2
        root = fsolve(func, x0)[0]
        roots.append(root)

    return roots


def print_with_errors(method_name, method_func, func, intervals, scipy_ref, eps):
    print(f"\n{method_name}:")

    manual_roots = (
        method_func(func, intervals, eps)
        if method_func != SimpleIteration
        else method_func(func, intervals, eps)
    )

    for i, (our_root, scipy_root) in enumerate(zip(manual_roots, scipy_ref)):
        if isinstance(our_root, str):
            print(f"x{i+1}: {our_root}")
        else:
            error = abs(our_root - scipy_root)
            print(
                f"x{i+1}: {our_root:.8f} | Погрешность: {error:.2e} | SciPy: {scipy_root:.8f}"
            )


rootIntervals1 = ClarifyRoots(Func1)
rootIntervals2 = ClarifyRoots(Func2)


scipyRoots1 = SciPyRoots(Func1, rootIntervals1)
scipyRoots2 = SciPyRoots(Func2, rootIntervals2)


print(f"\nУточненные интервалы:")
print(f"Func1: " + ", ".join(f"({a:.4f}, {b:.4f})" for a, b in rootIntervals1))
print(f"Func2: " + ", ".join(f"({a:.4f}, {b:.4f})" for a, b in rootIntervals2))
print("\n")
print_with_errors(
    "Func1: Дихотомия (eps=0.01)",
    DichotomyMethod,
    Func1,
    rootIntervals1,
    scipyRoots1,
    0.01,
)
print_with_errors(
    "Func2: Дихотомия (eps=0.01)",
    DichotomyMethod,
    Func2,
    rootIntervals2,
    scipyRoots2,
    0.01,
)
print("\n")
print_with_errors(
    "Func1: Простая итерация (eps=0.001)",
    SimpleIteration,
    IterProcess1,
    rootIntervals1,
    scipyRoots1,
    0.001,
)
print_with_errors(
    "Func2: Простая итерация (eps=0.001)",
    SimpleIteration,
    IterProcess2,
    rootIntervals2,
    scipyRoots2,
    0.001,
)
print("\n")
print_with_errors(
    "Func1: Метод Ньютона (eps=1e-6)",
    NewtonRoots,
    Func1,
    rootIntervals1,
    scipyRoots1,
    1e-6,
)
print_with_errors(
    "Func2: Метод Ньютона (eps=1e-6)",
    NewtonRoots,
    Func2,
    rootIntervals2,
    scipyRoots2,
    1e-6,
)

input()
