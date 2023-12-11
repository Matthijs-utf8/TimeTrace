import multiprocessing
import sys
from collections import deque

import numpy as np
import pyaudio
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget, QApplication
import pyqtgraph as pg
import re
import warnings
warnings.simplefilter("ignore")

from utils import get_microphone_sampling_frequency, write_audio, extract_peaks, guess_bph
from PyQTSettingsDialog import SettingsDialog
from MicrophoneThread import AudioProcessor
from PyQTCustomItems import CustomPushButton, CustomLabel, CustomSpinBox
from PyQTReportDialog import ReportWindow


class TimeTrace(QMainWindow):

    """
    A class for handling the main window of the TimeTrace application.

    This class extends QMainWindow and manages the UI and functionalities
    such as plotting audio data, saving audio, generating reports, and
    handling settings.

    Attributes:
        Various PyQt5 widgets and layout components.
    """

    def __init__(self):
        """Initializes the TimeTrace window with default settings and UI setup."""
        super().__init__()
        self.resize(1536, 864)  # Set width and height as desired
        self.setWindowTitle('TimeTrace')
        self.setWindowIcon(QIcon('./data/icons/audio-waves.png'))
        with open("./data/style.css", "r") as file: self.setStyleSheet(file.read()) # Read styles from sheet
        self.init_defaults()  # Initialize default settings and parameters
        self.init_ui()  # Initialize the user interface

    def init_defaults(self):

        """
        Initialize default settings for the application.

        Sets up various parameters related to audio processing, plotting,
        and application behavior.
        """

        # Settings
        self.liftangle = 52             # Lift angle
        self.movement_bph = "Auto"      # Movement BPH

        # Measurements
        self.rate = 0                 # Beats-per-second (s)
        self.beat_error = 0           # Beat error (ms)
        self.amplitude = 0            # Amplitude (-)

        # Buffer parameters
        self.archive_time = 60                                        # Length of archive (s)
        self.interval = 0.01                                          # Interval between reading of the live signal
        self.fs = get_microphone_sampling_frequency()                 # Sampling frequency
        self.sampling_factor = 10
        self.chunk_size = 0                                           # Size of the chunks

        # Reading, displaying and storing audio
        self.chunk_count = 0
        self.dotted_plot_seconds = 60
        self.archive_audio = deque(maxlen=self.fs * (self.archive_time))

    def init_ui(self):

        """
        Set up the User Interface for the TimeTrace application.

        This method constructs the layout, toolbar, buttons, labels, and
        plots for the main window.
        """

        # Define main layout of window
        layout = QHBoxLayout()
        self.plot_widget_layout = QVBoxLayout()

        """ Create all buttons """
        self.save_button = CustomPushButton("./data/icons/save-file.png", self.save_audio, "Save audio", size=35)
        self.report_button = CustomPushButton("./data/icons/report.png", self.write_report, "Write report", size=35)
        self.listen_button = CustomPushButton("./data/icons/listen-on.png", self.on_listen, "Start listening", size=35)
        self.whisper_button = CustomPushButton("./data/icons/whisper-on.png", self.on_whisper, "Start whispering", size=35)
        self.control_button = CustomPushButton("./data/icons/control.png", self.open_settings, "Open settings", size=35)

        """ Create all labels"""
        self.day_rate_label = CustomLabel(self.rate)
        self.day_rate_label.setPrefix("Rate: ")
        self.day_rate_label.setSuffix(" s/day")
        self.beat_error_label = CustomLabel(self.beat_error)
        self.beat_error_label.setPrefix("Beat error: ")
        self.beat_error_label.setSuffix(" ms")
        self.amplitude_label = CustomLabel(self.amplitude)
        self.amplitude_label.setPrefix("Amplitude: ")
        self.amplitude_label.setSuffix("")

        """ Construct toolbar """
        self.addToolBar(Qt.TopToolBarArea, self.create_toolbar([
                                                                self.save_button,
                                                                self.report_button,
                                                                self.listen_button,
                                                                self.whisper_button,
                                                                self.control_button,
                                                                ]))
        self.update_labels()

        """ Create a plot widget for the dotted plot """
        self.dotted_widget = pg.PlotWidget()
        self.dotted_widget.plotItem.setMouseEnabled(x=False)
        self.scatterplotitem = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(255, 240, 200, 120))
        self.dotted_widget.addItem(self.scatterplotitem)
        self.plot_widget_layout.addWidget(self.dotted_widget)

        """ Add live signal graph to window """
        self.live_widget = pg.PlotWidget()
        self.live_widget.plotItem.setMouseEnabled(x=False, y=False)
        self.live_widget.getPlotItem().getViewBox().setAutoVisible(y=False)
        self.live_widget.setYRange(-1, 1)
        self.live_widget.setFixedHeight(150)
        self.live_widget.getPlotItem().hideAxis('bottom')  # This line removes the x-axis
        self.live_data = self.live_widget.plot(pen=(255, 240, 200))
        self.live_data.setDownsampling(auto=False)
        self.live_data.setClipToView(True)
        self.plot_widget_layout.addWidget(self.live_widget)

        """ Add the final plot widget layout to the window """
        layout.addLayout(self.plot_widget_layout)

        """ Set central widget (necessary for PyQT5) """
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def create_toolbar(self, buttons):

        """
        Creates a toolbar for the main window.

        Args:
            buttons (list): A list of QPushButton objects to be added to the toolbar.

        Returns:
            QToolBar: The constructed toolbar with added buttons and features.
        """

        toolbar = self.addToolBar('Toolbar')
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        for index, button in enumerate(buttons):
            toolbar.addWidget(button)
            toolbar.addSeparator()

        """ Add movement bph spinbox"""
        self.bph_spinbox = CustomSpinBox()
        self.bph_spinbox.setMinimum(0)
        self.bph_spinbox.setMaximum(36000)
        self.bph_spinbox.setPrefix("BPH ")
        self.bph_spinbox.setSingleStep(1800)
        self.update_bph(self.movement_bph)
        self.bph_spinbox.valueChanged.connect(self.update_bph)
        self.bph_spinbox.setContentsMargins(0, 0, 0, 0)
        toolbar.addWidget(self.bph_spinbox)
        toolbar.addSeparator()

        """ Add lift angle spinbox"""
        self.liftangle_spinbox = CustomSpinBox()
        self.liftangle_spinbox.setPrefix("Lift Angle ")
        self.liftangle_spinbox.setMinimum(30)
        self.liftangle_spinbox.setMaximum(60)
        self.liftangle_spinbox.setValue(self.liftangle)
        self.liftangle_spinbox.valueChanged.connect(self.update_liftangle)
        self.liftangle_spinbox.setContentsMargins(0, 0, 0, 0)
        toolbar.addWidget(self.liftangle_spinbox)

        self.day_rate_label.setContentsMargins(50, 0, 0, 0)
        toolbar.addWidget(self.day_rate_label)
        self.beat_error_label.setContentsMargins(50, 0, 0, 0)
        toolbar.addWidget(self.beat_error_label)
        self.amplitude_label.setContentsMargins(50, 0, 0, 0)
        toolbar.addWidget(self.amplitude_label)

        return toolbar

    def save_audio(self):

        """
        Open a dialog to save the current audio buffer to a file.

        This method allows the user to select a file path and format
        for saving the current audio data.
        """

        # Open a file dialog to specify the save path and name
        options = QFileDialog.Options()
        file_dialog = QFileDialog()
        file_dialog.setOptions(options)

        # Get the save path and selected file format from the user
        save_path, _ = file_dialog.getSaveFileName(self, "Save as", "", "MP3 Files (*.mp3);;WAV Files (*.wav)",
                                                   options=options)

        if save_path:
            if len(self.archive_audio) > 0:
                write_audio(np.array(self.archive_audio), self.fs, save_path)

    def write_report(self):
        """Open a dialog for generating and saving a report."""
        self.report_dialog = ReportWindow(self)
        self.report_dialog.exec_()

    def on_listen(self):

        """
        Handle the listen button click event.

        This method starts or stops the audio processing thread and updates the UI accordingly.
        """


        try:

            # Set thread working or turn it off
            if not hasattr(self, "audio_processor_process"):

                """ Calibrate BPH """
                p = pyaudio.PyAudio()
                stream = p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=self.fs,
                                input=True,
                                frames_per_buffer=self.fs)

                # Read one second of data from the queue
                data = stream.read(self.fs, exception_on_overflow=True)
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Stop the stream
                stream.stop_stream()
                stream.close()
                p.terminate()

                # Calculate movement BPH
                self.movement_bph = guess_bph(audio_data, self.fs)
                # self.movement_bph = 21600
                self.update_labels()
                del stream, p, audio_data, data

                # Set chunk size and interval based on BPH
                self.chunk_size = 2 * int(round(self.fs / (self.movement_bph / 3600)))
                self.interval = 0.1 * self.chunk_size / self.fs
                self.live_widget.setXRange(0, int(round(self.chunk_size/10)))
                self.dotted_widget.setXRange(0, self.dotted_plot_seconds)
                self.dotted_widget.setYRange(0, (self.chunk_size/2) / self.fs)

                # Create audio processor on different core
                self.audio_queue = multiprocessing.Queue()
                self.toggle_whisper = multiprocessing.Event()
                self.audio_processor = AudioProcessor(self.audio_queue, self.toggle_whisper, self.chunk_size, self.fs)
                self.audio_processor_process = multiprocessing.Process(target=self.audio_processor.fetch_signal)
                self.audio_processor_process.daemon = True
                self.audio_processor_process.start()

                # Start feature extraction process
                self.feuture_extraction_queue = multiprocessing.Queue()
                self.labels_queue = multiprocessing.Queue()
                self.tics_queue = multiprocessing.Queue()
                self.feature_extraction_process = multiprocessing.Process(target=extract_peaks, args=(self.feuture_extraction_queue, self.fs, self.liftangle, self.labels_queue, self.tics_queue))
                self.feature_extraction_process.daemon = True
                self.feature_extraction_process.start()

                # Create QTimer objects for periodic updates of the graphs and labels
                self.timer1 = QTimer(self)
                self.timer1.timeout.connect(self.update_live_display_continuous)
                self.timer1.start(int(round(self.interval * 1000)))
                self.timer2 = QTimer(self)
                self.timer2.timeout.connect(self.update_labels_continuous)
                self.timer2.start(1000)
                self.timer3 = QTimer(self)
                self.timer3.timeout.connect(self.update_dotted_display_continuous)
                self.timer3.start(int(round(self.interval * 1000)))

                # Set icon
                self.listen_button.setIcon(QIcon("./data/icons/listen-off.png"))

            else:
                self.cleanup()
                self.listen_button.setIcon(QIcon("./data/icons/listen-on.png"))

        except Exception as e:
            print(f"Error in on_listen: {e}")

    def on_whisper(self):

        """
        Handle the whisper button click event.

        This method toggles the whisper mode and updates the button icon.
        """

        try:

            if self.toggle_whisper.is_set():
                self.toggle_whisper.clear()
                self.whisper_button.setIcon(QIcon("./data/icons/whisper-on.png"))
            else:
                self.toggle_whisper.set()
                self.whisper_button.setIcon(QIcon("./data/icons/whisper-off.png"))

        except Exception as e:
            print(f"Error in on_listen: {e}")

    def open_settings(self):
        """Open the settings dialog."""
        self.settings_dialog = SettingsDialog(self)
        self.settings_dialog.exec_()

    def update_live_display_continuous(self):
        """Continuously (try to) update the live audio display."""
        if not self.audio_queue.empty():
            # Read the audio, put it in the feature extraction queue and add it to the archive
            audio_data = self.audio_queue.get()
            self.feuture_extraction_queue.put(audio_data)
            self.archive_audio.extend(audio_data)
            self.update_live_display(audio_data=audio_data)
            self.chunk_count += 1

    def update_live_display(self, audio_data):

        """
        Update the live audio display with new data.

        Args:
            audio_data (ndarray): The new audio data to be displayed.
        """

        # Plot down-sampled audio to screen
        sampled_audio_data = audio_data[::self.sampling_factor]
        sampled_audio_data = sampled_audio_data / max(abs(sampled_audio_data))
        self.live_data.setData(
            sampled_audio_data,
            pen=(255, 240, 200)
        )

    def update_dotted_display_continuous(self):
        """Continuously (try to) update the dotted audio display."""
        if not self.tics_queue.empty():
            t, tic, tac = self.tics_queue.get()
            self.update_dotted_display(t, tic, tac)

    def update_dotted_display(self, t, tic, tac):

        """
        Update the dotted audio display with new data.

        Args:
            t (float): The current time.
            tic (ndarray): Array of three tic values.
            tac (ndarray): Array of three tac values.
        """

        if t > self.dotted_widget.plotItem.viewRange()[0][1] or t == 0:
            self.dotted_widget.clear()
            self.scatterplotitem = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(255, 240, 200, 200))
            self.dotted_widget.addItem(self.scatterplotitem)

            center = np.mean(tic) / self.fs
            self.dotted_widget.setYRange(center - (0.05 * self.chunk_size / self.fs),
                                         center + (0.05 * self.chunk_size / self.fs))
            self.dotted_widget.setXRange(t,
                                         t + self.dotted_plot_seconds)

            for rate in [0, 10, 30, 60, 120]:
                slope = (rate / 86400) * (self.chunk_size / self.fs)
                x_range = [t, t + self.dotted_plot_seconds*0.95]

                y_range = [center, center + slope * (x_range[1] - x_range[0])]
                self.dotted_widget.plot(x_range, y_range, pen=pg.mkPen((100, 170, 240, 120), width=1))
                label = pg.TextItem(text=f"{rate} s/day", color=(255, 240, 200), anchor=(0, 0.5))
                self.dotted_widget.addItem(label)
                label.setPos(x_range[1], y_range[1])

                if rate != 0:
                    y_range = [center, center - slope * (x_range[1] - x_range[0])]
                    self.dotted_widget.plot(x_range, y_range, pen=pg.mkPen((100, 170, 240, 120), width=1))
                    label = pg.TextItem(text=f"-{rate} s/day", color=(255, 240, 200), anchor=(0, 0.5))
                    self.dotted_widget.addItem(label)
                    label.setPos(x_range[1], y_range[1])

        self.scatterplotitem.addPoints(
                [{'pos': [t, np.mean(tic) / self.fs], 'data': 1},
                 {'pos': [t, (np.mean(tac) - (self.chunk_size / 2)) / self.fs], 'data': 1}]
            )

    def update_labels_continuous(self):
        """Continuously (try to) update the labels with new data."""
        if not self.labels_queue.empty():
            self.amplitude, self.rate, self.beat_error = self.labels_queue.get()
            self.update_labels()

    def update_labels(self):

        """Update the labels with current data values."""

        self.update_bph(self.movement_bph)
        if type(self.movement_bph) != str:
            self.bph_spinbox.setStyleSheet(re.sub(r'color: #?[0-9a-fA-F]+|color: rgb\(\d+,\s*\d+,\s*\d+\);', "color: rgb(100, 170, 240);", self.bph_spinbox.styleSheet()))
            self.day_rate_label.setText(self.rate)
        else:
            self.bph_spinbox.setStyleSheet(re.sub(r'color: #?[0-9a-fA-F]+|color: rgb\(\d+,\s*\d+,\s*\d+\);', "color: rgb(240, 120, 120);", self.bph_spinbox.styleSheet()))
            self.day_rate_label.setText(0)
        self.beat_error_label.setText(round(self.beat_error * 1000, 1))
        self.amplitude_label.setText(self.amplitude)

    def update_bph(self, value):

        """
        Update the beats per hour (BPH) value.

        Args:
            value (int or str): The new BPH value to be set.
        """

        self.movement_bph = value
        if (type(value) == str and value == "Auto") or value == 0:
            self.bph_spinbox.setSpecialValueText("BPH Auto")
        else:
            self.bph_spinbox.setSpecialValueText("")
            self.bph_spinbox.setValue(value)

    def update_liftangle(self, value):

        """
        Update the beats per hour (BPH) value.

        Args:
            value (int or str): The new BPH value to be set.
        """

        self.liftangle = value

    def cleanup(self):

        """Clean up resources and threads when the window is closed or listening stops."""

        try:
            if hasattr(self, "audio_processor_process"):
                self.audio_processor_process.terminate()
                self.audio_processor_process.join(timeout=1)  # Timeout to avoid hanging
                del self.audio_processor_process, self.audio_processor, self.audio_queue
        except Exception as e:
            print(f"Error during cleanup: {e}")
        try:
            if hasattr(self, "feature_extraction_process"):
                self.feature_extraction_process.terminate()
                self.feature_extraction_process.join(timeout=1)
                del self.feature_extraction_process, self.feuture_extraction_queue, self.labels_queue, self.tics_queue
        except Exception as e:
            print(f"Error during cleanup: {e}")
        try:
            if hasattr(self, "timer1"):
                self.timer1.stop()
                del self.timer1
        except Exception as e:
            print(f"Error during cleanup: {e}")
        try:
            if hasattr(self, "timer2"):
                self.timer2.stop()
                del self.timer2
        except Exception as e:
            print(f"Error during cleanup: {e}")
        try:
            if hasattr(self, "timer3"):
                self.timer3.stop()
                del self.timer3
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def closeEvent(self, event):

        """
        Handle the close event for the window.

        Args:
            event (QCloseEvent): The close event.
        """

        self.cleanup()
        event.accept()

def main():
    app = QApplication(sys.argv)
    main_window = TimeTrace()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()