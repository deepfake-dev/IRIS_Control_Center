import flet as ft
from supabase import create_client, Client
import os

def LoginPage(page: ft.Page):
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)
    
    async def push_to_ctrl():
        await page.push_route("/control")
    
    async def try_login(e):
        input_code = univ_code_tf.value
        input_password = password_tf.value
        
        login_button.disabled = True
        page.update()

        try:
            # STEP 1: Call the secure RPC function to fetch email
            rpc_response = supabase.rpc(
                "get_email_by_code", 
                {"p_user_code": input_code}
            ).execute()
            
            user_email = rpc_response.data
            
            if not user_email:
                dialog = ft.AlertDialog(
                    title=ft.Text("Cannot log in."),
                    content=ft.Text("User Code not found!"),
                    alignment=ft.Alignment.CENTER,
                    title_padding=ft.Padding.all(25),
                )
                page.show_dialog(dialog)
                enable_login(None)
                return

            # STEP 2: Authenticate using the retrieved email
            auth_response = supabase.auth.sign_in_with_password({
                "email": user_email,
                "password": input_password
            })
            
            # Save the authentication tokens so other pages can use them
            page.session.store.set("access_token", auth_response.session.access_token)
            page.session.store.set("refresh_token", auth_response.session.refresh_token)

            # STEP 3: Verify the user is from ICT using the new secure RPC
            ict_response = supabase.rpc(
                "check_if_ict", 
                {"p_user_code": input_code}
            ).execute()
            
            # The RPC returns True, False, or None
            is_ict = True if ict_response.data is True else False

            if is_ict:
                await push_to_ctrl()
            else:
                # Clean up session since they aren't authorized for this app
                supabase.auth.sign_out()
                page.session.store.remove("access_token")
                page.session.store.remove("refresh_token")
                
                dialog = ft.AlertDialog(
                    title=ft.Text("Access Denied."),
                    content=ft.Text("Only ICT can access the Control Center!"),
                    alignment=ft.Alignment.CENTER,
                    title_padding=ft.Padding.all(25),
                )
                page.show_dialog(dialog)
                enable_login(None)

        except Exception as ex:
            print(ex)
            error_message = str(ex)
            display_message = "Please check your credentials!"
            
            if "Invalid login credentials" in error_message:
                display_message = "Incorrect Password!"
                
            dialog = ft.AlertDialog(
                title=ft.Text("Cannot log in."),
                content=ft.Text(display_message),
                alignment=ft.Alignment.CENTER,
                title_padding=ft.Padding.all(25),
            )
            page.show_dialog(dialog)
            enable_login(None)

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