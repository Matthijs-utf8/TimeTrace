from PyQt5.QtWidgets import QVBoxLayout, QDialog
from PyQTCustomItems import CustomCheckBox

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle('Settings')
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        """ Settings """

        # Create the custom checkbox with text and an initial state (checked or not)
        self.plot_checkbox = CustomCheckBox(text="live Plot", initial_checked=True)
        self.plot_checkbox.stateChanged.connect(self.toggle_plot_visibility)
        layout.addWidget(self.plot_checkbox)
        self.setLayout(layout)

    def toggle_plot_visibility(self, state):
        if state:
            self.parent.live_widget.show()
        else:
            self.parent.live_widget.hide()

