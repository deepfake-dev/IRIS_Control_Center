import flet as ft
from login_page import LoginPage
from control_page import ControlPage
from dotenv import load_dotenv

load_dotenv()

async def main(page: ft.Page):
    page.title = "IRIS Control Center"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 1280
    page.window.height = 800
    # page.window.resizable = False
    # page.window.maximizable = False
    page.views[0].padding = ft.Padding.all(0)

    def route_change():
        page.views.clear()
        
        if page.route == "/":
            page.views.append(LoginPage(page))
        elif page.route == "/control":
            page.views.append(ControlPage(page))
            
        page.update()

    async def view_pop(view):
        page.views.pop()
        top_view = page.views[-1]
        await page.push_route(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop

    # page.route = "/control"
    
    route_change()
    await page.window.center()

ft.run(main)