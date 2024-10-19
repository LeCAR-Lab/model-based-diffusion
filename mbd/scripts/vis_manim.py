import numpy as np
from manim import *
import os


class PlotGraph(Scene):
    def construct(self):
        render = True
        dt = 1 / 15
        caption_size = 30

        # Create title "Model-Free Diffusion"
        title = Text("Standard Model-Free Diffusion", font_size=40).shift(UP * 3)

        # Create two box side by side
        box1 = Rectangle(width=4, height=4).shift(LEFT * 3)
        box2 = Rectangle(width=4, height=4).shift(RIGHT * 3)

        # Create circle in the center of each box filled with blue
        r = 0.5
        circle1 = Circle(radius=r, color=BLUE, fill_opacity=1).shift(box1.get_center())
        circle2 = Circle(radius=r, color=BLUE, fill_opacity=1).shift(box2.get_center())

        # Create curve from [0, -1.5] to [0, 1.5] passing [-0.5, 0]
        demo_curves = []
        for _ in range(4):
            x = np.random.uniform(0.5, 0.8)
            curve = ParametricFunction(
                lambda t: np.array([x * np.cos(t / 3.0 * np.pi), t, 0]),
                t_range=[-1.5, 1.5],
                color=RED,
            ).shift(box1.get_center())
            demo_curves.append(curve)
        for _ in range(4):
            x = np.random.uniform(0.5, 0.8)
            curve = ParametricFunction(
                lambda t: np.array([-x * np.cos(t / 3.0 * np.pi), t, 0]),
                t_range=[-1.5, 1.5],
                color=RED,
            ).shift(box1.get_center())
            demo_curves.append(curve)

        # Create vector field in the box2
        def demo_vector_field_fn(x):
            value = (x[0] / 0.6) ** 2 + (x[1] / 1.5) ** 2 - 1.0
            vec = 0.2 * np.array([x[0] / 0.6, x[1] / 1.5])
            if value < 0:
                return vec
            else:
                return -vec

        demo_vector_field = ArrowVectorField(
            demo_vector_field_fn,
            x_range=[-2, 2],
            y_range=[-2, 2],
            length_func=lambda norm: np.clip(norm, 0, 0.5),
            color=RED,
        ).shift(box2.get_center())

        demo_title = Text("Demonstrations", font_size=30).shift(
            box1.get_top() + UP * 0.5
        )
        diffusion_title = Text("Diffusion Process", font_size=30).shift(
            box2.get_top() + UP * 0.5
        )

        caption1 = Text(
            "Standard diffusion learns score (→) only from demonstrations.",
            font_size=caption_size,
        ).shift(DOWN * 3)

        # Add the box to the scene
        if render:
            # create box1, box2, circle1, circle2, title
            self.play(
                Create(box1),
                Create(box2),
                Create(circle1),
                Create(circle2),
                Create(title),
                run_time=2.0,
            )
            # create demo_curves
            self.play(
                *[Create(curve) for curve in demo_curves],
                Create(demo_title),
                run_time=2.0,
            )
            # create caption
            self.play(Write(caption1), run_time=1.0)
            # create demo_vector_field
            self.play(
                TransformFromCopy(demo_curves[0], demo_vector_field),
                Create(diffusion_title),
                run_time=3.0,
            )
        else:
            self.add(
                box1,
                box2,
                circle1,
                circle2,
                title,
                *demo_curves,
                demo_vector_field,
                demo_title,
                diffusion_title,
                caption1,
            )

        # create scatter plot with color changing
        betas = np.linspace(1e-3, 1e-1, 60)
        # betas = np.linspace(0.3, 0.3, 3)
        alphas = 1.0 - betas
        trajs = []
        traj_scatters = []
        for scale in [1.0, 1.3, -1.1, -1.25]:
            ys = np.linspace(-1.5, 1.5, 20)
            xs = scale * 0.5 * np.cos(ys / 3.0 * np.pi)
            # add noise
            xys = np.stack([xs, ys], axis=1)
            xyss = [xys]
            for alpha in alphas:
                xys = xys * np.sqrt(alpha) + np.random.randn(20, 2) * np.sqrt(1 - alpha)
                xys = np.clip(xys, -1.9, 1.9)
                xyss.append(xys)
            xyss = xyss[::-1]
            scatter = []
            for i, (x, y) in enumerate(xyss[0]):
                c = interpolate_color(WHITE, RED, i / 20)
                scatter.append(
                    Dot(np.array([x, y, 0]), color=c).shift(box2.get_center())
                )
            self.add(*scatter)
            traj_scatters.append(scatter)
            trajs.append(xyss)
        for t in range(alphas.shape[0] + 1):  # diffusion step
            for i in range(len(traj_scatters)):  # each trajectory
                scatter = traj_scatters[i]
                xys = trajs[i][t]
                for j, (x, y) in enumerate(xys):
                    scatter[j].move_to(np.array([x, y, 0]) + box2.get_center())
            if render:
                self.wait(dt)

        # make circle 2 scale up 1.8
        caption2 = Text(
            "When the constraints/model changes (e.g., the obstacle is larger), \nthe score cannot adapt accordingly.",
            font_size=caption_size,
        ).shift(DOWN * 3)
        r_scale = 1.8
        r_new = r * r_scale
        if render:
            self.play(Transform(caption1, caption2))
            self.pause(1.0)
            self.play(
                circle2.animate.scale(r_scale),
                circle1.animate.scale(r_scale),
                run_time=2.0,
            )
        else:
            circle1.scale(r_scale)
            circle2.scale(r_scale)
            # self.remove(caption1)
            self.add(caption2)

        # update trajs
        # create scatter plot with color changing
        caption4 = Text(
            "Here, the trajectory would hit the obstacle given the new constraint.",
            font_size=caption_size,
        ).shift(DOWN * 3)
        trajs_new = []
        for scale in [1.0, 1.3, -1.1, -1.25]:
            ys = np.linspace(-1.5, 1.5, 20)
            xs = scale * 0.5 * np.cos(ys / 3.0 * np.pi)
            mask = ys > -r_new * np.cos(np.pi / 6)
            right_mask = np.logical_and(mask, xs > 0)
            left_mask = np.logical_and(mask, xs < 0)
            xs[right_mask] = r_new * np.sin(np.pi / 6)
            xs[left_mask] = -r_new * np.sin(np.pi / 6)
            ys[mask] = -r_new * np.cos(np.pi / 6)
            # add noise
            xys = np.stack([xs, ys], axis=1)
            xyss = [xys]
            for alpha in alphas:
                xys = xys * np.sqrt(alpha) + np.random.randn(20, 2) * np.sqrt(1 - alpha)
                xys = np.clip(xys, -1.9, 1.9)
                xyss.append(xys)
            xyss = xyss[::-1]
            trajs_new.append(xyss)
        if render:
            self.play(Transform(caption1, caption4))
        else:
            self.remove(caption2)
            self.add(caption4)
        for t in range(alphas.shape[0] + 1):  # diffusion step
            for i in range(len(traj_scatters)):  # each trajectory
                scatter = traj_scatters[i]
                xys = trajs_new[i][t]
                for j, (x, y) in enumerate(xys):
                    scatter[j].move_to(np.array([x, y, 0]) + box2.get_center())
            if render:
                self.wait(dt)

        if render:
            self.wait(1.0)

        # change title to "Model-Based Diffusion"
        title2 = Text("Model-Based Diffusion", font_size=40).shift(UP * 3)
        for scatter in traj_scatters:
            for s in scatter:
                s.set_fill(opacity=0)
        if render:
            self.play(
                Transform(title, title2),
                circle1.animate.scale(1 / r_scale),
                circle2.animate.scale(1 / r_scale),
                run_time=1.5,
            )
        else:
            self.add(title2)
            circle1.scale(1 / r_scale)
            circle2.scale(1 / r_scale)
            # remove scatter plot
            # for scatter in traj_scatters:
            #     self.remove(*scatter)

        # add latex equation $$\dot{x} = \frac{u}{m}, \\ \text{s.t.} \quad \|x\| \geq r$$
        model_title = Text("Model", font_size=30).shift(box1.get_top() + UP * 0.5)
        eq = MathTex(r"\dot{x} = u,  \quad \text{s.t. } \|x\|_2 \geq 0.5").shift(
            box1.get_bottom() + DOWN * 1.0
        )
        caption3 = Text(
            "MBD overcomes this by \ncomputing score with model",
            font_size=caption_size,
        ).shift(box2.get_bottom() + DOWN * 1.0)
        if render:
            # remove caption
            self.play(Transform(caption1, caption3))
            # transform demo_curves to eq
            self.play(
                Transform(demo_title, model_title),
                TransformFromCopy(circle1, eq),
                *[FadeOut(curve) for curve in demo_curves],
                run_time=2.0,
            )
            self.wait(1.0)
        else:
            self.add(eq)
            self.remove(*demo_curves)
            self.add(model_title)
            # self.remove(caption4)
            self.add(caption3)
            # self.play(Transform(caption1, caption3))

        # update vector field
        scale = 0.4

        def demo_vector_field_fn(x):
            value = (x[0] / 0.6 / scale) ** 2 + (x[1] / 1.5) ** 2 - 1.0
            vec = 0.2 * np.array([x[0] / 0.6 / scale, x[1] / 1.5])
            if value < 0:
                return vec
            else:
                return -vec

        small_vector_field = ArrowVectorField(
            demo_vector_field_fn,
            x_range=[-2, 2],
            y_range=[-2, 2],
            length_func=lambda norm: np.clip(norm, 0.2, 0.5),
            color=RED,
        ).shift(box2.get_center())
        eq_new = MathTex(r"\dot{x} = u,  \quad \text{s.t. } \|x\|_2 \geq 0.2").shift(
            box1.get_bottom() + DOWN * 1.0
        )
        if render:
            self.play(
                Transform(demo_vector_field, small_vector_field),
                Transform(eq, eq_new),
                circle1.animate.scale(scale),
                circle2.animate.scale(scale),
                run_time=2.0,
            )
        else:
            self.add(small_vector_field)
            self.remove(demo_vector_field)
            circle1.scale(scale)
            circle2.scale(scale)

        eq_new = MathTex(
            r"\dot{x} = u,  \quad \text{s.t. } \|x\|_{\infty} \geq 0.5"
        ).shift(box1.get_bottom() + DOWN * 1.0)

        def square_vector_field_fn(x):
            value = np.abs(x[0]) < 0.6 and np.abs(x[1]) < 0.6
            vec = 0.2 * np.array([x[0] / 1.0 / scale, x[1] / 1.5])
            if value:
                return vec
            else:
                return -vec

        square_vector_field = ArrowVectorField(
            square_vector_field_fn,
            x_range=[-2, 2],
            y_range=[-2, 2],
            length_func=lambda norm: np.clip(norm, 0.2, 0.5),
            color=RED,
        ).shift(box2.get_center())
        square1 = Square(side_length=1.0, color=BLUE, fill_opacity=1).shift(
            box1.get_center()
        )
        square2 = Square(side_length=1.0, color=BLUE, fill_opacity=1).shift(
            box2.get_center()
        )
        if render:
            self.play(
                Transform(demo_vector_field, square_vector_field),
                Transform(eq, eq_new),
                # circle1.animate.scale(scale/0.4),
                # circle2.animate.scale(scale/0.4),
                Transform(circle1, square1),
                Transform(circle2, square2),
                run_time=2.0,
            )
        else:
            # self.add(large_vector_field)
            self.remove(small_vector_field)
            circle1.scale(scale / 0.4)
            circle2.scale(scale / 0.4)

        scale = 1.8
        eq_new = MathTex(r"\dot{x} = u,  \quad \text{s.t. } \|x\|_2 \geq 0.9").shift(
            box1.get_bottom() + DOWN * 1.0
        )
        large_vector_field = ArrowVectorField(
            demo_vector_field_fn,
            x_range=[-2, 2],
            y_range=[-2, 2],
            length_func=lambda norm: np.clip(norm, 0.2, 0.5),
            color=RED,
        ).shift(box2.get_center())
        circle1 = Circle(radius=0.9, color=BLUE, fill_opacity=1).shift(
            box1.get_center()
        )
        circle2 = Circle(radius=0.9, color=BLUE, fill_opacity=1).shift(
            box2.get_center()
        )
        if render:
            self.play(
                Transform(square1, circle1),
                Transform(square2, circle2),
                Transform(demo_vector_field, large_vector_field),
                Transform(eq, eq_new),
                run_time=2.0,
            )
        else:
            self.add(large_vector_field)
            self.remove(small_vector_field)
            circle1.scale(scale / 0.4)
            circle2.scale(scale / 0.4)

        trajs_new = []
        for s in [1.0, 1.3, -1.1, -1.25]:
            ys = np.linspace(-1.5, 1.5, 20)
            xs = s * scale * 0.5 * np.cos(ys / 3.0 * np.pi)
            # add noise
            xys = np.stack([xs, ys], axis=1)
            xyss = [xys]
            for alpha in alphas:
                xys = xys * np.sqrt(alpha) + np.random.randn(20, 2) * np.sqrt(1 - alpha)
                xys = np.clip(xys, -1.9, 1.9)
                xyss.append(xys)
            xyss = xyss[::-1]
            trajs_new.append(xyss)
        for t in range(alphas.shape[0] + 1):  # diffusion step
            if t == 0:
                for scatter in traj_scatters:
                    for s in scatter:
                        s.set_fill(opacity=1)
            for i in range(len(traj_scatters)):  # each trajectory
                scatter = traj_scatters[i]
                xys = trajs_new[i][t]
                for j, (x, y) in enumerate(xys):
                    scatter[j].move_to(np.array([x, y, 0]) + box2.get_center())
            if render:
                self.wait(dt)

        title3 = Text(
            "Model-Based Diffusion Applied to High-dim Control Tasks", font_size=40
        ).shift(UP * 3)
        if render:
            self.play(Transform(title, title3))
            self.wait(6.0)
        else:
            self.add(title3)

        # # Create vector field in the box2
        # def demo_vector_field_fn(x):
        #     scale = 1.8
        #     value = (x[0] / 0.6 / scale) ** 2 + (x[1] / 1.5) ** 2 - 1.0
        #     vec = 0.2 * np.array([x[0] / 0.6 / scale, x[1] / 1.5])
        #     if value < 0:
        #         return vec
        #     else:
        #         return -vec

        # model_vector_field = ArrowVectorField(
        #     demo_vector_field_fn,
        #     x_range=[-2, 2],
        #     y_range=[-2, 2],
        #     length_func=lambda norm: np.clip(norm, 0, 0.5),
        #     color=RED,
        # ).shift(box2.get_center())
        # eq_new = MathTex(r"\dot{x} = u,  \quad \text{s.t. } \|x\|_2 \geq 0.9").shift(
        #     box1.get_bottom() + DOWN * 1.0
        # )
        # if render:
        #     self.play(Transform(demo_vector_field, model_vector_field), Transform(eq, eq_new))
        # else:
        #     self.add(model_vector_field)
        #     self.remove(demo_vector_field)


if __name__ == "__main__":
    module_name = "vis_manim"
    path = f"{module_name}.py"
    # config to 1080p 60fps
    os.system(f"manim {path} -pql -r 1920,1080 --fps 60")
