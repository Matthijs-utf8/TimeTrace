from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QSpinBox,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QWidget,
    QHBoxLayout,
    QLineEdit,
)


class CustomCheckBoxLineEdit(QWidget):
    """
    A custom widget consisting of a checkbox and a line edit.

    Args:
        label_text (str): The label text for the checkbox.
        text (str): The initial text for the line edit.
    """

    def __init__(self, label_text="", text=""):
        super().__init__()

        # Initialize variables
        self.label = label_text
        self.text = text

        # Main layout
        layout = QHBoxLayout()

        # Create a checkbox
        self.checkbox = CustomCheckBox(label_text, initial_checked=True)
        self.checkbox.setMinimumWidth(300)
        layout.addWidget(self.checkbox)

        # Line edit
        self.line_edit = QLineEdit()
        self.line_edit.setText(str(text))
        layout.addWidget(self.line_edit)

        # Set layout
        self.setLayout(layout)

    def setText(self, text):
        self.line_edit.setText(str(text))
        self.text = text


class CustomCheckBox(QWidget):
    """
    A custom checkbox widget.

    Args:
        text (str): The label text for the checkbox.
        initial_checked (bool): The initial checked state.
    """

    stateChanged = pyqtSignal(bool)

    def __init__(self, text='', initial_checked=False):
        super().__init__()
        self.checked = initial_checked

        # Create layout for checkbox and text
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignLeft)

        # Create label for the checkmark icon
        self.checkmark_label = QLabel()
        self.checkmark_label.setAlignment(Qt.AlignLeft)

        # Create label for the text
        self.text_label = QLabel(text)
        self.text_label.setAlignment(Qt.AlignLeft)

        # Set initial icon
        self.update_icon()

        # Add labels to layout
        layout.addWidget(self.checkmark_label)
        layout.addWidget(self.text_label)

        # Set layout for the widget
        self.setLayout(layout)

    def update_icon(self):
        if self.checked:
            # Blue checkmark icon
            pixmap = QPixmap(20, 20)
            pixmap.fill(QColor("transparent"))
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor("#64aaf0"), 2, Qt.SolidLine))
            painter.drawRect(2, 2, 16, 16)
            painter.drawLine(5, 10, 8, 15)
            painter.drawLine(8, 15, 15, 5)
            painter.end()
            self.checkmark_label.setPixmap(pixmap)
        else:
            # Red outline icon
            pixmap = QPixmap(20, 20)
            pixmap.fill(QColor("transparent"))
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor("#f07878"), 2, Qt.SolidLine))
            painter.drawRect(2, 2, 16, 16)
            painter.end()
            self.checkmark_label.setPixmap(pixmap)

    def isChecked(self):
        return self.checked

    def mousePressEvent(self, event):
        self.checked = not self.checked
        self.update_icon()
        self.stateChanged.emit(self.checked)

    def setText(self, text):
        self.text_label.setText(text)


class CustomSpinBox(QSpinBox):
    """
    A custom integer spin box widget.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set the initial properties
        self.setStyleSheet("""

            * {
                background-color: transparent;
                color: #fff4c9;  /* Change the text color */
                font-family: "Courier", sans-serif;
                font-weight: bold; /* Make the font bold */
                font-size: 20px; /* Change the font size to 10 */
            }
            QSpinBox {
                background-color: transparent;
                border: 1px solid black;
                color: #fff4c9;
            }

            QSpinBox::down-button, QSpinBox::up-button {
                background-color: transparent;
                border-radius: 0;
                color: fff4c9;
            }

            QSpinBox::up-arrow {
                width: 12px;
                height: 12px;
                image: url(../Data/icons/up-arrow.png);
                image-position: center center;
            }

            QSpinBox::down-arrow {
                width: 12px;
                height: 12px;
                image: url(../Data/icons/down-arrow.png);
                image-position: center center;
            }
        """)

        self.setButtonSymbols(QSpinBox.UpDownArrows)
        self.setCorrectionMode(QSpinBox.CorrectToNearestValue)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(0, 35)
        self.setMaximumSize(4000, 35)
        self.setFont(QFont("Courier", 10))
        self.setInputMethodHints(Qt.ImhDigitsOnly)
        self.setWrapping(False)
        self.setFrame(False)


class CustomDoubleSpinBox(QDoubleSpinBox):
    """
    A custom double spin box widget.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set the initial properties
        self.setStyleSheet("""

            * {
                background-color: transparent;
                color: #fff4c9;  /* Change the text color */
                font-family: "Courier", sans-serif;
                font-weight: bold; /* Make the font bold */
                font-size: 20px; /* Change the font size to 10 */
            }
            QDoubleSpinBox {
                background-color: transparent;
                border: 1px solid black;
                color: #fff4c9;
            }

            QDoubleSpinBox::down-button, QDoubleSpinBox::up-button {
                background-color: transparent;
                border-radius: 0;
                color: fff4c9;
            }

            QDoubleSpinBox::up-arrow {
                width: 12px;
                height: 12px;
                image: url(../Data/icons/up-arrow.png);
                image-position: center center;
            }

            QDoubleSpinBox::down-arrow {
                width: 12px;
                height: 12px;
                image: url(../Data/icons/down-arrow.png);
                image-position: center center;
            }
        """)

        self.setButtonSymbols(QSpinBox.UpDownArrows)
        self.setCorrectionMode(QSpinBox.CorrectToNearestValue)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(0, 35)
        self.setMaximumSize(4000, 35)
        self.setFont(QFont("Courier", 10))
        self.setInputMethodHints(Qt.ImhDigitsOnly)
        self.setWrapping(False)
        self.setFrame(False)


class CustomLabel(QLabel):
    """
    A custom label widget.

    Args:
        text (str): The text to display in the label.
        parent: The parent widget (optional).
    """

    def __init__(self, text, parent=None):
        super().__init__(str(text), parent)
        self._prefix = ""
        self._suffix = ""
        self.setContentsMargins(0, 0, 0, 0)

    def setText(self, text):
        super().setText(f"{self._prefix}{str(text)}{self._suffix}")

    def setPrefix(self, prefix):
        self._prefix = prefix
        self.setText(f"{self._prefix}{str(self.text())}{self._suffix}")

    def setSuffix(self, suffix):
        self._suffix = suffix
        self.setText(f"{self._prefix}{str(self.text())}{self._suffix}")


class CustomPushButton(QPushButton):
    """
    A custom push button widget.

    Args:
        icon (str): The path to the button's icon.
        function: The function to call when the button is clicked.
        tooltip (str): The tooltip text for the button.
        size (int): The size of the button (both width and height).
    """

    def __init__(self, icon, function, tooltip, size=50):
        super().__init__()
        self.setIcon(QIcon(icon))
        self.setFixedSize(size, size)
        self.setIconSize(QSize(int(round(size * 0.8)), int(round(size * 0.8))))
        self.clicked.connect(function)
        self.setToolTip(tooltip)

    def enterEvent(self, event):
        # Change cursor to pointing hand when mouse enters the button
        self.setCursor(Qt.PointingHandCursor)

    def leaveEvent(self, event):
        # Reset cursor to default when mouse leaves the button
        self.setCursor(Qt.ArrowCursor)

