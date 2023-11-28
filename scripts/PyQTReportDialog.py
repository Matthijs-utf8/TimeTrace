from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QDateTime, QStandardPaths
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyQTCustomItems import CustomCheckBoxLineEdit, CustomPushButton
from typing import Tuple, List, Optional

class ReportWindow(QDialog):
    """
    A dialog window for generating and saving a PDF report.

    Args:
        app: The parent application.

    Attributes:
        author_input (CustomCheckBoxLineEdit): Input field for the author's name.
        watch_input (CustomCheckBoxLineEdit): Input field for the watch information.
        movement_input (CustomCheckBoxLineEdit): Input field for the movement information.
        serial_input (CustomCheckBoxLineEdit): Input field for the serial number.
        liftangle_input (CustomCheckBoxLineEdit): Input field for the lift angle.
        movement_bph_input (CustomCheckBoxLineEdit): Input field for beats per hour.
        rate_input (CustomCheckBoxLineEdit): Input field for rate.
        beat_error_input (CustomCheckBoxLineEdit): Input field for beat error.
        amplitude_input (CustomCheckBoxLineEdit): Input field for amplitude.
        generate_button (CustomPushButton): Button to generate and save the report.
        file_dialog (QFileDialog): File dialog for selecting the save location.
    """

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setFixedWidth(800)
        self.setWindowIcon(QIcon('../Data/icons/audio-waves.png'))
        self.setStyleSheet(open("../Data/style.css").read())
        self.min_label_width = 300
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Report Window")
        layout = QVBoxLayout()

        # Create line edits for report information
        self.author_input = CustomCheckBoxLineEdit("Author:", "")
        self.load_user_name()
        self.author_input.line_edit.textChanged.connect(self.save_user_name)
        self.watch_input = CustomCheckBoxLineEdit("Watch:", "")
        self.watch_input.line_edit.textChanged.connect(self.watch_input.setText)
        self.movement_input = CustomCheckBoxLineEdit("Movement:", "")
        self.movement_input.line_edit.textChanged.connect(self.movement_input.setText)
        self.serial_input = CustomCheckBoxLineEdit("Serial Number:", "")
        self.serial_input.line_edit.textChanged.connect(self.serial_input.setText)
        self.liftangle_input = CustomCheckBoxLineEdit("Lift Angle:", self.app.liftangle)
        self.movement_bph_input = CustomCheckBoxLineEdit("BPH:", self.app.movement_bph)
        self.rate_input = CustomCheckBoxLineEdit("Rate:", self.app.rate)
        self.beat_error_input = CustomCheckBoxLineEdit("Beat Error:", self.app.beat_error)
        self.amplitude_input = CustomCheckBoxLineEdit("Amplitude:", self.app.amplitude)

        layout.addWidget(QLabel("Report Information"))
        layout.addWidget(self.author_input)
        layout.addWidget(self.watch_input)
        layout.addWidget(self.movement_input)
        layout.addWidget(self.serial_input)
        layout.addWidget(QLabel("Measurement Information"))
        layout.addWidget(self.liftangle_input)
        layout.addWidget(self.movement_bph_input)
        layout.addWidget(self.rate_input)
        layout.addWidget(self.beat_error_input)
        layout.addWidget(self.amplitude_input)

        # Create the save button
        self.generate_button = CustomPushButton("../Data/icons/save-report.png", self.generate_report, "Save report", size=35)
        self.generate_button.setFixedWidth(780)
        layout.addWidget(self.generate_button)

        # Initialize file dialog
        self.file_dialog = QFileDialog()

        # Set layout
        self.setLayout(layout)

    def generate_report(self):
        """
        Generate and save the PDF report.
        """
        self.generate_button.setEnabled(False)
        default_file_name = QStandardPaths.standardLocations(QStandardPaths.DownloadLocation)[0] + '\\untitled'
        file_path, _ = self.file_dialog.getSaveFileName(self, "Save Report", default_file_name, "PDF Files (*.pdf)")

        if not file_path:
            self.generate_button.setEnabled(True)
            return

        c = canvas.Canvas(file_path, pagesize=letter)
        y = self.create_pdf_header(c)
        self.create_pdf_measurements(c, y_position=y)
        self.create_pdf_statement(c)
        self.create_pdf_image(c)

        c.save()
        self.generate_button.setEnabled(True)

    def create_pdf_header(self, c: canvas.Canvas, y_position: int = 750, x_position: int = 50) -> int:
        """
        Create the header section of the PDF report.

        Args:
            c (canvas.Canvas): The PDF canvas.
            y_position (int): The starting y-position for drawing.

        Returns:
            int: The updated y-position after drawing the header.
        """
        c.setFont("Courier-Bold", 20)
        c.drawString(x_position, y_position, "Accuracy Report")
        y_position -= 25
        c.setFont("Courier-Bold", 14)
        c.drawString(x_position, y_position, "Report information")
        y_position -= 18
        c.setFont("Courier", 12)
        c.drawString(x_position + 150, y_position, QDateTime.currentDateTime().toString("yyyy-MM-dd") + " " + QDateTime.currentDateTime().toString("hh:mm:ss"))
        y_position -= 15

        info = [(_input.label, _input.text) for _input in [self.author_input,
                                                           self.watch_input,
                                                           self.movement_input,
                                                           self.serial_input]
                if _input.checkbox.isChecked()]

        for label, value in info:
            c.setFont("Courier-Bold", 12)
            c.drawString(x_position, y_position, label)
            c.setFont("Courier", 12)
            c.drawString(x_position + 150, y_position, value)
            y_position -= 15
        c.line(x_position, y_position, 600 - (min(x_position, 50)), y_position)  # Replace coordinates as needed
        y_position -= 15

        return y_position

    def create_pdf_measurements(self, c: canvas.Canvas, y_position: int, x_position: int = 50):
        """
        Create the measurement information section of the PDF report.

        Args:
            c (canvas.Canvas): The PDF canvas.
            y_position (int): The starting y-position for drawing.
            x_position (int): The x-position for drawing.

        Returns:
            int: The updated y-position after drawing the measurement information.
        """
        c.setFont("Courier-Bold", 14, leading=0)
        c.drawString(x_position, y_position, "Measurement information")
        y_position -= 18

        info = [(_input.label, _input.text) for _input in
                [
                    self.movement_bph_input,
                    self.rate_input,
                    self.beat_error_input,
                    self.amplitude_input,
                    self.liftangle_input
                ]
                if _input.checkbox.isChecked()
        ]

        for label, value in info:
            c.setFont("Courier-Bold", 12)
            c.drawString(x_position, y_position, label)
            c.setFont("Courier", 12)
            c.drawString(x_position + 300, y_position, str(value))
            y_position -= 15

    def create_pdf_statement(self, c: canvas.Canvas):
        """
        Create the statement section of the PDF report.

        Args:
            c (canvas.Canvas): The PDF canvas.
        """
        c.setFont("Courier", 5)
        statement = [
            "This report was generated with TimeTrace.",
            "The information in this report is provided as is, without warranty of any kind.",
            "TimeTrace is not responsible for any inaccuracies or errors in this report.",
        ]
        y_position = 23
        for line in statement:
            c.drawString(67, y_position, line)
            y_position -= 4

    def create_pdf_image(self, c: canvas.Canvas):
        """
        Create and add an image to the PDF report.

        Args:
            c (canvas.Canvas): The PDF canvas.
        """
        image_path = '../Data/icons/audio-waves.png'
        c.drawImage(image_path, 50, 15, width=11, height=11)  # Adjust the coordinates and size as needed

    def save_user_name(self):
        """
        Save the user's name to a text file.
        """
        user_name = self.author_input.text
        if user_name:
            with open("user_name.txt", "w") as file:
                file.write(user_name)

    def load_user_name(self) -> Optional[str]:
        """
        Load the user's name from a text file.

        Returns:
            Optional[str]: The user's name if found, or None if the file does not exist.
        """
        try:
            with open("user_name.txt", "r") as file:
                user_name = file.read()
                if user_name:
                    self.author_input.setText(user_name)
                    return user_name
        except FileNotFoundError:
            pass
