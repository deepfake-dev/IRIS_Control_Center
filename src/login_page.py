import flet as ft

def LoginPage(page: ft.Page):
    async def push_to_ctrl():
        await page.push_route("/control")
    
    async def try_login(e):
        # result = await login(univ_code_tf.value, password_tf.value)
        result = True
        if result:
            await push_to_ctrl()
        else:
            dialog = ft.AlertDialog(
                title=ft.Text("Cannot log in."),
                content=ft.Text("Please check your credentials!"),
                alignment=ft.Alignment.CENTER,
                title_padding=ft.Padding.all(25),
            )

            page.show_dialog(dialog)
    
    def enable_login(e):
        if all([
            univ_code_tf.value,
            univ_code_tf.value != "",
            password_tf.value,
            password_tf.value != ""
        ]):
            login_button.disabled = False
        else:
            login_button.disabled = True
        
        login_button.update()

    univ_code_tf = ft.TextField(
        label="Enter your employee code",
        expand=True,
        border_color=ft.Colors.WHITE_54,
        on_change=enable_login
    )

    password_tf = ft.TextField(
        label="Enter your password",
        password=True,
        expand=True,
        border_color=ft.Colors.WHITE_54,
        on_change=enable_login
    )

    login_button = ft.FilledButton(
        "Login",
        width=128,
        disabled=True,
        on_click=try_login,
        style=ft.ButtonStyle(
            bgcolor={
                ft.ControlState.DISABLED: ft.Colors.GREY,
                ft.ControlState.HOVERED: "#e4c6fa",
                ft.ControlState.DEFAULT: "#cd9ef7",
            },
            color={
                ft.ControlState.DISABLED: ft.Colors.WHITE,
                ft.ControlState.DEFAULT: "#452981",
            }
        )
    )
    
    container = ft.Container(
        width=640,
        height=400,
        bgcolor="#333333",
        border=ft.Border.all(0.1, ft.Colors.WHITE_54),
        border_radius=16,
        alignment=ft.Alignment.CENTER,
        padding=ft.Padding.all(16),
        content=ft.Column(
            alignment=ft.MainAxisAlignment.SPACE_AROUND,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Row(
                    spacing=16,
                    alignment=ft.MainAxisAlignment.CENTER,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Image(src="icon.png", width=128, height=128),
                        ft.Text(
                            "IRIS System\n", 
                            weight=ft.FontWeight.W_300, 
                            size=72,
                            spans=[
                                ft.TextSpan("Sign in to your Account", style=ft.TextStyle(size=24, weight=ft.FontWeight.W_900)),
                            ]
                        )
                    ]
                ),
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=32),
                    content = ft.Column(
                        spacing=16,
                        controls=[
                            univ_code_tf,
                            password_tf
                        ]
                    ),
                ),
                login_button
            ]
        )
    )

    return ft.View(
        route="/",
        bgcolor="#262626",
        padding=0,
        controls=[
            ft.SafeArea(
                expand=True,
                content=ft.Container(
                    content=container,
                    alignment=ft.Alignment.CENTER,
                )
            )
        ]
    )