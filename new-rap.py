def f(x):
    return 4*x**3 - 15*x**2 + 16*x - 4

def f_prime(x):
    return 12*x**2 - 30*x + 16

def newton_raphson(x0, n):
    xn = x0
    for _ in range(n):
        fxn = f(xn)
        fpxn = f_prime(xn)
        xn = xn - fxn / fpxn
    return xn


x0 = 0  # 초기값
n = int(input("반복 횟수: ")) #반복할 횟수 입력

# 뉴턴랩슨법을 사용하여 Xn 계산
for i in range(1, n+1):
    xn = newton_raphson(x0, i)
    print(f"x_{i} = {xn:.30f}")
