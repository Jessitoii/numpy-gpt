"""
PyQt5-based chat interface for interacting with the trained NanoGPT model.

This module provides a modern, dark-themed GUI that allows for interactive 
chatting and continuous text generation with the Transformer model.
- Left panel: Mode selection and controls.
- Center panel: Message history with chat bubbles.
- Bottom panel: Text input.
"""

import os
import threading

from PyQt5 import QtCore, QtGui, QtWidgets

import tokenizer
from saving import load_model
from main import DeepNanoGPT
from utils import generate


# Initialize CuPy or NumPy fallback from the tokenizer module
cp = tokenizer.cp
MAX_SEQ_DEFAULT = 128


class TokenGeneratorWorker(QtCore.QThread):
    """
    Worker thread for non-blocking text generation.

    This worker runs the model's generation process in a background thread
    to ensure the PyQt5 event loop remains responsive during inference.

    Signals:
        token_ready (str): Emitted whenever a new character is generated.
        finished: Emitted when the generation process completes.
    """
    token_ready = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, model, prompt, max_tokens=128, temperature=1.0, continuous=False, stop_event=None, parent=None):
        """
        Initializes the generator worker.

        Args:
            model (DeepNanoGPT): The Transformer model instance.
            prompt (str): The initial text to start generation from.
            max_tokens (int): Maximum number of tokens to generate per cycle.
            temperature (float): Sampling temperature.
            continuous (bool): If True, continues generating until stopped.
            stop_event (Event): Thread-safe event to signal termination.
            parent (QObject): Parent object for the thread.
        """
        super().__init__(parent)
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.continuous = continuous
        self.stop_event = stop_event or threading.Event()

    def run(self):
        """
        Executes the generation loop.
        """
        try:
            while True:
                # Generate a chunk of characters
                # Note: We iterate over characters produced by the generate utility
                for token in generate(self.model, self.prompt, length=64, temperature=self.temperature):
                    if self.stop_event.is_set():
                        return
                    # Send token to the UI thread
                    self.token_ready.emit(token)
                    
                # Exit if not in continuous mode
                if not self.continuous:
                    break
        finally:
            self.finished.emit()


class ChatWindow(QtWidgets.QMainWindow):
    """
    The main application window for the NanoGPT Chat.

    Handles UI layout, themes, user input, and model lifecycle management.
    """

    def __init__(self):
        """
        Initializes the chat window and loads the Transformer model.
        """
        super().__init__()
        self.setWindowTitle("NanoGPT Chat")
        self.resize(1100, 720)

        # Threading and model state
        self.stop_event = threading.Event()
        self.worker = None
        self.active_reply_label = None
        
        # Initialize the deep model with standard hyperparameters
        self.model = DeepNanoGPT(
            vocab_size=tokenizer.vocab_size,
            embed_size=256,
            seq_len=128,
            num_blocks=4,
            num_heads=8,
        )
        
        # Load pre-trained weights
        load_model(
            model=self.model,
            filename="./models/final_model/whatsapp_gpt.pkl",
        )
        
        self.setup_ui()
        self.apply_dark_theme()
        self.update_buttons()

    def setup_ui(self):
        """
        Builds the main layout structure.
        """
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Assemble the panels
        self.left_panel = self.build_left_panel()
        self.chat_panel = self.build_chat_panel()

        root_layout.addWidget(self.left_panel)
        root_layout.addWidget(self.chat_panel, 1)

    def build_left_panel(self):
        """
        Creates the sidebar with mode selection and controls.

        Returns:
            QFrame: The constructed side panel.
        """
        frame = QtWidgets.QFrame()
        frame.setFixedWidth(220)
        frame.setObjectName("sidePanel")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(16)

        title = QtWidgets.QLabel("Mode")
        title.setObjectName("sideTitle")
        layout.addWidget(title)

        # Radio buttons for interaction mode
        self.mode_group = QtWidgets.QButtonGroup(frame)
        self.mode_continuous = QtWidgets.QRadioButton("Continuous Generation")
        self.mode_chat = QtWidgets.QRadioButton("Chat Mode")
        self.mode_chat.setChecked(True)
        
        for btn in (self.mode_continuous, self.mode_chat):
            btn.toggled.connect(self.update_buttons)
            self.mode_group.addButton(btn)
            layout.addWidget(btn)

        layout.addSpacing(12)
        
        # Action buttons
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setObjectName("dangerButton")
        
        self.start_btn.clicked.connect(self.start_continuous_mode)
        self.stop_btn.clicked.connect(self.stop_generation)
        
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        layout.addStretch(1)

        # Model information label
        info = QtWidgets.QLabel("Model: NanoGPT\nGeneration runs in a QThread to avoid UI blocking.")
        info.setWordWrap(True)
        info.setObjectName("sideInfo")
        layout.addWidget(info)

        return frame

    def build_chat_panel(self):
        """
        Creates the main chat interface.

        Returns:
            QFrame: The constructed chat panel.
        """
        panel = QtWidgets.QFrame()
        panel.setObjectName("chatPanel")
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setContentsMargins(24, 20, 24, 20)
        vbox.setSpacing(12)

        header = QtWidgets.QLabel("NanoGPT Chat")
        header.setObjectName("chatTitle")
        vbox.addWidget(header)

        # Scrollable message area
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        container = QtWidgets.QWidget()
        self.messages_layout = QtWidgets.QVBoxLayout(container)
        self.messages_layout.setContentsMargins(0, 0, 0, 0)
        self.messages_layout.setSpacing(12)
        # Add stretch to push messages to the top
        self.messages_layout.addStretch(1)
        
        self.scroll.setWidget(container)
        vbox.addWidget(self.scroll, 1)

        # Input row
        input_row = QtWidgets.QHBoxLayout()
        input_row.setSpacing(10)
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Type a message...")
        self.input.returnPressed.connect(self.on_send)
        
        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.clicked.connect(self.on_send)
        
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.send_btn)
        vbox.addLayout(input_row)

        return panel

    def apply_dark_theme(self):
        """
        Applies GPT-style dark theme styling using QPalette and CSS.
        """
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(22, 22, 28))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(28, 28, 36))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(232, 232, 240))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(232, 232, 240))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(36, 36, 46))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(232, 232, 240))
        self.setPalette(palette)

        self.setStyleSheet(
            """
            QWidget { color: #e8e8f0; font-size: 14px; }
            #sidePanel { background: #111118; border-right: 1px solid #202028; }
            #chatPanel { background: #16161f; }
            #sideTitle, #chatTitle { font-size: 16px; font-weight: 600; }
            #sideInfo { color: #a0a0b5; font-size: 12px; }
            QLineEdit { background: #1f1f2b; border: 1px solid #2a2a38; border-radius: 8px; padding: 10px; }
            QPushButton { background: #2b2b3c; border: 1px solid #3a3a4a; border-radius: 8px; padding: 10px 14px; }
            QPushButton:hover { border-color: #5b8def; color: #dfe6ff; }
            QPushButton:disabled { background: #242432; color: #666; }
            QPushButton#dangerButton { background: #3a1e25; border-color: #a43a4a; color: #ffced6; }
            QScrollArea { border: none; }
            QRadioButton::indicator { width: 16px; height: 16px; }
            QRadioButton { spacing: 8px; }
            """
        )

    def update_buttons(self):
        """
        Enables/Disables buttons based on the current application state.
        """
        is_continuous = self.mode_continuous.isChecked()
        self.start_btn.setEnabled(is_continuous)
        self.stop_btn.setEnabled(is_continuous and self.worker is not None)
        self.send_btn.setEnabled(not is_continuous)
        self.input.setEnabled(True)

    def add_message(self, text, sender="user"):
        """
        Adds a chat bubble message to the message feed.

        Args:
            text (str): The message content.
            sender (str): Either "user" or "model".

        Returns:
            QLabel: The label containing the message text.
        """
        bubble = QtWidgets.QLabel(text)
        bubble.setWordWrap(True)
        bubble.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        bubble.setAlignment(QtCore.Qt.AlignLeft)
        bubble.setObjectName("msgBubble")

        wrapper = QtWidgets.QFrame()
        wrapper_layout = QtWidgets.QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        
        # Right align user messages, left align model messages
        if sender == "user":
            wrapper_layout.addStretch(1)
            wrapper_layout.addWidget(bubble, 0)
        else:
            wrapper_layout.addWidget(bubble, 0)
            wrapper_layout.addStretch(1)

        bubble.setStyleSheet(
            "background: #2c2c3a; border: 1px solid #353545; border-radius: 14px; padding: 12px;"
            if sender == "user"
            else "background: #1f2738; border: 1px solid #33415c; border-radius: 14px; padding: 12px;"
        )

        # Insert before the bottom stretch
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, wrapper)
        # Scroll down after the UI updates
        QtCore.QTimer.singleShot(0, self.scroll_to_bottom)
        return bubble

    def scroll_to_bottom(self):
        """
        Scrolls the chat view to the latest message.
        """
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def start_continuous_mode(self):
        """
        Begins continuous text generation.
        """
        if not self.mode_continuous.isChecked():
            return
        if self.worker is not None:
            return

        prompt = self.input.text().strip() or "Alperitto:"
        self.active_reply_label = self.add_message("", sender="model")
        
        self.stop_event.clear()
        self.worker = TokenGeneratorWorker(
            self.model,
            prompt=prompt,
            max_tokens=64,
            temperature=1.0,
            continuous=True,
            stop_event=self.stop_event,
        )
        self.worker.token_ready.connect(self.append_token)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        self.update_buttons()

    def stop_generation(self):
        """
        Signals the worker thread to stop generation.
        """
        if self.worker:
            self.stop_event.set()
        self.update_buttons()

    def on_send(self):
        """
        Handles user message submission in chat mode.
        """
        if self.mode_continuous.isChecked():
            return
        text = self.input.text().strip()
        if not text:
            return
            
        self.input.clear()
        self.add_message(text, sender="user")
        self.active_reply_label = self.add_message("", sender="model")

        self.stop_event.clear()
        self.worker = TokenGeneratorWorker(
            self.model,
            prompt=text,
            max_tokens=128,
            temperature=1.0,
            continuous=False,
            stop_event=self.stop_event,
        )
        self.worker.token_ready.connect(self.append_token)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        self.update_buttons()

    def append_token(self, token):
        """
        Updates the UI with a newly generated token.

        Args:
            token (str): The generated character.
        """
        if self.active_reply_label is None:
            self.active_reply_label = self.add_message("", sender="model")
        self.active_reply_label.setText(self.active_reply_label.text() + token)
        self.scroll_to_bottom()

    def on_worker_finished(self):
        """
        Callback for when the background generation thread completes.
        """
        self.worker = None
        self.update_buttons()


def main():
    """
    Main entry point for the GUI application.
    """
    import sys
    # Flag to prevent multiple model initializations if main.py is imported
    os.environ.setdefault("NANOGPT_UI", "1")
    app = QtWidgets.QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
