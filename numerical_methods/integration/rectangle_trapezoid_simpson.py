import numpy as np
import matplotlib.pyplot as plt


def Func1(x):
    return 1 / np.sqrt(0.2 * (x**2) + 1)


def Func2(x):
    return np.sqrt(x + 1) * np.log(x + 3)


def IntegrateRectangle(x, func, step):
    res = 0
    for i in range(1, x.size):
        res += func((x[i - 1] + x[i]) / 2)

    res *= step

    return res


def IntegrateTrapezoid(x, func, step):
    res = 0
    for i in range(1, x.size):
        res += func(x[i - 1]) + func(x[i])

    res *= step / 2

    return res


def IntegrateSimpson(f_x, step):
    res = 0
    x0 = f_x[0]
    xn = f_x[-1]
    x_even = [j for i, j in enumerate(f_x) if i > 0 and i % 2 == 0]
    x_even.pop(-1)
    x_odd = [j for i, j in enumerate(f_x) if i % 2 != 0]

    x_even_sum = sum(x_even) * 2
    x_odd_sum = sum(x_odd) * 4

    res = (step / 3) * (x0 + x_odd_sum + x_even_sum + xn)

    return res


def Integrate(x0, xn, n, func, method):
    x0, xn = x0, xn
    h = (xn - x0) / n
    x = np.arange(x0, xn, h, dtype=np.float64)
    x = np.append(x, xn)
    f_x = np.array([func(i) for i in x])

    if method == "rectangle":
        res = IntegrateRectangle(x, func, h)
    elif method == "trapezoid":
        res = IntegrateTrapezoid(x, func, h)
    elif method == "simpson":
        res = IntegrateSimpson(f_x, h)

    return res


def GetFinitDiffs(results_dict, method):
    values = results_dict[method]
    n_values = sorted(values.keys())

    first_differences = {}
    for i in range(1, len(n_values)):
        n = n_values[i]
        prev_n = n_values[i - 1]
        first_differences[n] = values[n] - values[prev_n]

    second_differences = {}
    for i in range(2, len(n_values)):
        n = n_values[i]
        prev_n = n_values[i - 1]
        second_differences[n] = first_differences[n] - first_differences[prev_n]

    third_differences = {}
    for i in range(3, len(n_values)):
        n = n_values[i]
        prev_n = n_values[i - 1]
        third_differences[n] = second_differences[n] - second_differences[prev_n]

    return {
        "values": values,
        "first_diff": first_differences,
        "second_diff": second_differences,
        "third_diff": third_differences,
    }


def PrintFinitDiffs(diff_data, method_name, func_name):
    print(f"\n{func_name}, {method_name}")
    print("|   n   |    I      |    ΔI      |    Δ²I     |    Δ³I     |")
    print("|-------|-----------|------------|------------|------------|")

    values = diff_data["values"]
    first_diff = diff_data["first_diff"]
    second_diff = diff_data["second_diff"]
    third_diff = diff_data["third_diff"]

    n_values = sorted(values.keys())

    print(f"|  10   | {values[10]:.6f}  |     —      |     —      |     —      |")

    if 20 in first_diff:
        print(
            f"|  20   | {values[20]:.6f}  | {first_diff[20]:+.6f}  |     —      |     —      |"
        )

    if 40 in second_diff:
        print(
            f"|  40   | {values[40]:.6f}  | {first_diff[40]:+.6f}  | {second_diff[40]:+.6f}  |     —      |"
        )

    if 80 in third_diff:
        print(
            f"|  80   | {values[80]:.6f}  | {first_diff[80]:+.6f}  | {second_diff[80]:+.6f}  | {third_diff[80]:+.6f}  |"
        )


n = [10, 20, 40, 80]
x01, xn1 = 1.3, 2.5
x02, xn2 = 0.15, 0.63


resFunc1 = {"rectangle": {}, "trapezoid": {}, "simpson": {}}
resFunc2 = {"rectangle": {}, "trapezoid": {}, "simpson": {}}


for n in n:
    resFunc1["rectangle"][n] = Integrate(x01, xn1, n, Func1, "rectangle")
    resFunc1["trapezoid"][n] = Integrate(x01, xn1, n, Func1, "trapezoid")
    resFunc1["simpson"][n] = Integrate(x01, xn1, n, Func1, "simpson")

    resFunc2["rectangle"][n] = Integrate(x02, xn2, n, Func2, "rectangle")
    resFunc2["trapezoid"][n] = Integrate(x02, xn2, n, Func2, "trapezoid")
    resFunc2["simpson"][n] = Integrate(x02, xn2, n, Func2, "simpson")


diff_func1_rect = GetFinitDiffs(resFunc1, "rectangle")
diff_func1_trap = GetFinitDiffs(resFunc1, "trapezoid")
diff_func1_simp = GetFinitDiffs(resFunc1, "simpson")

diff_func2_rect = GetFinitDiffs(resFunc2, "rectangle")
diff_func2_trap = GetFinitDiffs(resFunc2, "trapezoid")
diff_func2_simp = GetFinitDiffs(resFunc2, "simpson")


PrintFinitDiffs(diff_func1_rect, "Rectangle", "Func1")
PrintFinitDiffs(diff_func1_trap, "Trapezoid", "Func1")
PrintFinitDiffs(diff_func1_simp, "Simpson", "Func1")
print("\n")
PrintFinitDiffs(diff_func2_rect, "Rectangle", "Func2")
PrintFinitDiffs(diff_func2_trap, "Trapezoid", "Func2")
PrintFinitDiffs(diff_func2_simp, "Simpson", "Func2")


input()
