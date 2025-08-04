import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class WaveEquationSolver:
    def __init__(self, N, l, T, M, Nt, lambda_reg):
        self.N = N
        self.l, self.T = l, T
        self.M, self.Nt = M, Nt
        self.h, self.tau = np.array([lj / Mj for lj, Mj in zip(l, M)]), T / Nt
        self.lambda_reg = lambda_reg

    def solve_wave_equation_forward(self, u):
        N, M, Nt = self.N, self.M, self.Nt
        h, tau = self.h, self.tau

        # Первые итерации (n = 0, 1)
        y = np.zeros((N, Nt + 1, np.max(M) + 1))

        constant = np.sum(1 / h)
        coefficients = (tau / h) ** 2

        # Итерации 2 <= n <= Nt
        for n in range(2, Nt + 1):
            y[:, n, 0] = u[:, n]
            for j in range(N):
                y[j, n, 1:M[j]] = 2 * y[j, n-1, 1:M[j]] - y[j, n-2, 1:M[j]] + \
                    coefficients[j] * (y[j, n-1, 2:M[j]+1] - 2 *
                                       y[j, n-1, 1:M[j]] + y[j, n-1, :M[j]-1])
            for j in range(N):
                y[j][n][M[j]] = sum(
                    y[j][n][M[j]-1] / h[j] for j in range(N)) / constant

        return y

    def solve_wave_equation_backward(self, v):
        N, M, Nt = self.N, self.M, self.Nt
        h, tau = self.h, self.tau

        # Первая итерация (n = Nt)
        psi = np.zeros((N, Nt + 1, np.max(M) + 1))

        # Вторая итерация (n = Nt - 1)
        psi[:, Nt-1, :] = v * tau

        constant = np.sum(1 / h)
        coefficients = (tau / h) ** 2

        # Оставшиеся итерации (0 <= n <= Nt - 2)
        for n in range(Nt - 2, -1, -1):
            psi[:, n, 0] = 0
            for j in range(N):
                psi[j, n, 1:M[j]] = 2 * psi[j, n+1, 1:M[j]] - psi[j, n+2, 1:M[j]] + \
                    coefficients[j] * (psi[j, n+1, 2:M[j]+1] - 2 *
                                       psi[j, n+1, 1:M[j]] + psi[j, n+1, :M[j]-1])
            for j in range(N):
                psi[j][n][M[j]] = sum(
                    psi[j][n][M[j]-1] / h[j] for j in range(N)) / constant

        return psi

    def scalar_H(self, u, v):
        tau = self.tau
        return sum(np.dot(u[j], v[j]) * tau for j in range(self.N))

    def scalar_F(self, u, v):
        h = self.h
        return sum(np.dot(u[j], v[j]) * h[j] for j in range(self.N))

    def J(self, u, f):
        y = self.solve_wave_equation_forward(u)
        v = y[:, self.Nt, :] - f
        J1 = self.scalar_F(v, v)
        J2 = self.scalar_H(u, u) * self.lambda_reg
        return J1 + J2

    def grad_J(self, u, f):
        y = self.solve_wave_equation_forward(u)
        v = y[:, self.Nt, :] - f

        psi = self.solve_wave_equation_backward(v)
        psi_x = (psi[:, :, 1] - psi[:, :, 0]) / self.h[:, np.newaxis]

        gradient1 = 2 * psi_x
        gradient2 = 2 * self.lambda_reg * u
        return gradient1 + gradient2
    
    def optimize(self, u0, f, eps=0.000400, max_iter=100):
        gamma = 0.1
        u_old = u0.copy()
        J_old = self.J(u_old, f)
        for i in range(max_iter):
            gradient = self.grad_J(u_old, f)
            u_new = u_old - gamma * gradient
            J_new = self.J(u_new, f)
            error = abs(J_new - J_old) / J_old
            if error < eps:
                break
            u_old = u_new
            J_old = J_new
        return u_new

    def show_animation(self, y, f, u_optimal, u_precise, speed=1):
        speed = 1 / speed
        N, M = self.N, self.M
        l = self.l

        fig, axes = plt.subplots(N, 2, figsize=(10, 10))
        if N == 1:
            axes = np.array([axes])
        plt.subplots_adjust(bottom=0.2, top=0.9, wspace=0.4, hspace=0.6)

        lines = []
        f_lines = []
        time_texts = []
        opt_lines = []
        precise_lines = []

        y_min = min(np.min(y), np.min(f))
        y_max = max(np.max(y), np.max(f))
        u_min = min(np.min(u_optimal), np.min(u_precise))
        u_max = max(np.max(u_optimal), np.max(u_precise))

        for i in range(N):
            ax = axes[i, 0]
            line, = ax.plot([], [], 'r-', linewidth=2)
            lines.append(line)

            x_i = np.linspace(0, l[i], M[i] + 1)
            f_line, = ax.plot(x_i, f[i][0:M[i] + 1], 'b--', linewidth=0.5)
            f_lines.append(f_line)

            time_text = ax.text(0.02, 0.8, '', transform=ax.transAxes)
            time_texts.append(time_text)

            ax.set_xlim(0, l[i])
            ax.set_ylim(y_min - 0.1 * abs(y_max - y_min),
                        y_max + 0.1 * abs(y_max - y_min))
            ax.set_title(f'Решение $y_{i+1}(T, x)$')
            ax.set_xlabel(f'$x$')
            ax.set_ylabel(f'$y$')

            ax_opt = axes[i, 1]

            opt_line, = ax_opt.plot([], [], 'g-', linewidth=2, label='Approximate')
            opt_lines.append(opt_line)

            precise_line, = ax_opt.plot([], [], 'm--', linewidth=2, label='Precise')
            precise_lines.append(precise_line)

            ax_opt.set_xlim(0, self.T)
            ax_opt.set_ylim(u_min - 0.1 * abs(u_max - u_min),
                            u_max + 0.1 * abs(u_max - u_min))
            ax_opt.set_title(f'Управление $u_{i+1}(t)$')
            ax_opt.set_xlabel('$t$')
            ax_opt.set_ylabel(f'$u$')
            ax_opt.legend(loc='lower left')

        def init():
            for line in lines + opt_lines + precise_lines:
                line.set_data([], [])
            for time_text in time_texts:
                time_text.set_text('')
            return lines + f_lines + time_texts + opt_lines + precise_lines

        def animate(t):
            global restart_clicked, anim
            for i, line in enumerate(lines):
                x_i = np.linspace(0, l[i], M[i] + 1)
                y_i = y[i][t, :][0:M[i] + 1]
                line.set_data(x_i, y_i)

            for i, opt_line in enumerate(opt_lines):
                u_i = u_optimal[i, :t+1]
                t_i = np.linspace(0, t * self.tau, t + 1)
                opt_line.set_data(t_i, u_i)

            for i, precise_line in enumerate(precise_lines):
                u_precise_i = u_precise[i, :t+1]
                t_i = np.linspace(0, t * self.tau, t + 1)
                precise_line.set_data(t_i, u_precise_i)

            if t == Nt:
                anim.event_source.stop()
                restart_clicked = False
            return lines + f_lines + time_texts + opt_lines + precise_lines

        global anim
        anim = FuncAnimation(fig, animate, init_func=init, frames=self.Nt + 1,
                             interval=int(self.T * 1000 / self.Nt * speed), blit=True)

        plt.show()


N = 3
l = np.array([10, 10, 10])
T = 10
Nt = 1000

# C_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
C_list = [0.9]
for C in C_list:
    tau = T / Nt
    M = np.array(list(map(int, l / (tau / C))))
    lambda_reg = 0.00001
    anim, restart_clicked = None, None

    x = np.array([np.linspace(0, lj, num=np.max(M)+1) for lj in l])
    f1 = np.sin(x[0])
    f2 = np.sin(x[1])
    f3 = np.sin(x[2])
    f = np.array([f1, f2, f3])

    solver = WaveEquationSolver(N, l, T, M, Nt, lambda_reg)
    t = np.array([np.linspace(0, T, num=Nt+1)])
    u0 = np.zeros((N, Nt + 1))
    u_optimal = solver.optimize(u0, f)
    
    u_1 = u_2 = u_3 = np.sin(10 - t)
    u_precise = np.vstack(np.array([u_1, u_2, u_3]))
    u_dif = u_optimal - u_precise
    dif = np.sqrt(solver.scalar_H(u_dif[:, 2:], u_dif[:, 2:]) / solver.scalar_H(u_precise[:, 2:], u_precise[:, 2:])) * 100
    print(f'Отклонение: {dif:.3f} при Nt = {Nt}, C = {C}, M = {M[0]}')

    y = solver.solve_wave_equation_forward(u_optimal)
    solver.show_animation(y, f, u_optimal, u_precise, speed=20)
