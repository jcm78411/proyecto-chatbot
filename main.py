import queue
import asyncio
import flet as ft
import threading
from modules.chat_logic import chat_rag
from modules.tts_module import _speak, stop_speaking

voice_enabled = True


def main(page: ft.Page):
    page.title = "Chat con IA"
    page.window.width = 620
    page.window.height = 700
    page.window.center()
    # page.window.max_width = 620
    # page.window.max_height = 700
    # page.window.maximizable = False
    # page.window.resizable = False
    page.update()

    # ✅ ListView con autoscroll
    chat_display = ft.ListView(expand=True, spacing=10, padding=10, auto_scroll=True)

    user_input = ft.TextField(label="Mensaje", expand=True, autofocus=True)
    send_btn = ft.ElevatedButton("Enviar", icon=ft.Icons.SEND)
    clear_btn = ft.ElevatedButton("Limpiar chat", icon=ft.Icons.DELETE)

    # 🌀 Spinner flotante
    overlay = ft.Container(
        visible=False,
        bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
        alignment=ft.alignment.center,
        content=ft.Column(
            [
                ft.ProgressRing(width=70, height=70, stroke_width=5),
                ft.Text(
                    "🤖 Pensando...", color=ft.Colors.WHITE, size=18, weight="bold"
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        expand=True,
    )

    def add_message(text, is_user=True):
        """Agrega un mensaje con estilo tipo WhatsApp (burbujas asimétricas)."""
        if is_user:
            bubble_color = ft.Colors.LIGHT_BLUE_100
            alignment = ft.MainAxisAlignment.END
            border = ft.border_radius.only(
                top_left=15, top_right=15, bottom_left=15, bottom_right=0
            )
        else:
            bubble_color = ft.Colors.GREY_300
            alignment = ft.MainAxisAlignment.START
            border = ft.border_radius.only(
                top_left=15, top_right=15, bottom_left=0, bottom_right=15
            )

        chat_display.controls.append(
            ft.Row(
                [
                    ft.Container(
                        content=ft.Text(
                            text,
                            color=ft.Colors.BLACK,
                            selectable=True,
                            size=14,
                        ),
                        bgcolor=bubble_color,
                        padding=10,
                        border_radius=border,
                        margin=ft.margin.only(left=10, right=10),
                        width=400,
                        shadow=ft.BoxShadow(blur_radius=2, spread_radius=0.5),
                    )
                ],
                alignment=alignment,
            )
        )
        chat_display.update()

    # 🔊 Cola para manejar los textos que se reproducen por voz
    tts_queue = queue.Queue()

    def tts_worker():
        """Reproduce los mensajes de la IA en orden."""
        global voice_enabled
        while True:
            texto = tts_queue.get()
            if texto is None:
                break
            if voice_enabled:
                _speak(texto)
            tts_queue.task_done()

    # 🧵 Inicia el hilo del TTS
    threading.Thread(target=tts_worker, daemon=True).start()

    # 🔇 Función para silenciar/reanudar voz
    def toggle_voice(e):
        global voice_enabled
        voice_enabled = not voice_enabled

        if not voice_enabled:
            stop_speaking()  # 🔇 Detiene voz actual inmediatamente

        voice_btn.text = "🔊 Reanudar voz" if not voice_enabled else "🔇 Silenciar voz"
        page.update()

    # 🔘 Botón de control de voz
    voice_btn = ft.ElevatedButton(
        "🔇 Silenciar voz", icon=ft.Icons.VOLUME_UP, on_click=toggle_voice
    )

    async def send_message(e):
        text = user_input.value.strip()
        if not text:
            return

        # Mostrar mensaje del usuario
        add_message(text, is_user=True)
        user_input.value = ""
        overlay.visible = True
        page.update()

        # 🧠 Ejecutar la lógica de IA sin bloquear la UI
        respuesta_ia = await asyncio.to_thread(chat_rag, text)

        # Mostrar respuesta
        add_message(respuesta_ia, is_user=False)
        overlay.visible = False
        page.update()

        # 🗣️ Enviar texto a la cola para lectura por voz
        tts_queue.put(respuesta_ia)

    def clear_chat(e):
        chat_display.controls.clear()
        page.update()

    # Eventos de botones y campo de texto
    send_btn.on_click = send_message
    clear_btn.on_click = clear_chat
    user_input.on_submit = send_message

    # 🧩 Contenido principal con overlay flotante
    main_content = ft.Column(
        [
            ft.Text(
                "🤖 Bienvenido al Chat IA",
                size=22,
                weight="bold",
                text_align="center",
            ),
            ft.Container(
                chat_display,
                height=500,
                width=600,
                bgcolor=ft.Colors.GREY_100,
                border_radius=10,
                padding=10,
            ),
            ft.Row([user_input, ft.Column([send_btn, clear_btn, voice_btn])]),
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # 🔹 Stack para superponer el overlay
    page.add(
        ft.Stack(
            [
                main_content,
                overlay,  # Se muestra solo cuando overlay.visible = True
            ]
        )
    )

if __name__ == "__main__":
    ft.app(target=main)
