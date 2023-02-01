import pandas as pd
from PyQt5.QtWidgets import QApplication, QTabWidget, QFileDialog
from UI import Ui_TabWidget
from mp_select_data_4R_7341  import MainSelectDataProgram


class MainWindow(QTabWidget, Ui_TabWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_TabWidget()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pathSelect.clicked.connect(self.open_file)
        self.ui.QCnumberInput.selectionChanged.connect(self.ui.QCnumberInput.clear)
        self.ui.calibration.clicked['bool'].connect(self.ui.QCnumberInput.setDisabled)
        self.ui.QC.clicked['bool'].connect(self.ui.QCnumberInput.setEnabled)
        self.ui.calibration.clicked['bool'].connect(self.ui.QCnumberInput.clear)
        self.ui.cleanPathButton.clicked.connect(self.clean_context)
        self.ui.cleanPathButton.clicked.connect(self.ui.progressBar.reset)
        self.ui.runButton.clicked.connect(self.get_parameter)
        self.ui.runButton.clicked.connect(self.run_program)
        
    def open_file(self):
        self.filename_list, _ = QFileDialog.getOpenFileNames(self, "Open file") # start path
        
        for index, filename in enumerate(self.filename_list):
            self.ui.showPath.insertPlainText(str(index+1) + ' ' + filename.split('/')[-1] + '\n')

    def clean_context(self):
        self.ui.showPath.clear()

    def get_parameter(self):

        self.parameter = {}
        
        # File Path List
        self.parameter['path_list'] = self.filename_list

        # Calibration and QC
        self.parameter['calibration'] = self.ui.calibration.isChecked()
        self.parameter['QC'] = self.ui.QC.isChecked()
        self.parameter['number'] = self.ui.QCnumberInput.text()

        # Select Data
        self.parameter['whether_select_data'] = self.ui.selectYesButton.isChecked()

        # Basic Parameter
        self.parameter['analysis_channel'] = self.ui.channelInput.currentText()
        self.parameter['cylinder_concentration'] = self.ui.cylinderInput.text()
        self.parameter['datapoint_number'] = self.ui.selectInput.text()
        self.parameter['buffer'] = self.ui.bufferInput.text()
        self.parameter['remove_spike'] = self.ui.spikeLabel.isChecked()        
        self.parameter['remove_spike_threshold'] = self.ui.thresholdInput.text()

        # MFC Max Flow Rate Parameter
        self.parameter['chemical_MFC_max_flow_rate'] = self.ui.chemicalMaxInput.text()
        self.parameter['chemical_dilute_MFC_max_flow_rate'] = self.ui.chemicalDiluteMaxInput.text()
        self.parameter['humidity_MFC_max_flow_rate'] = self.ui.humMaxInput.text()
        self.parameter['humidity_dilute_MFC_max_flow_rate'] = self.ui.humDiluteMaxInput.text()

        # MFC Channel Setting
        self.parameter['chemical_MFC_channel'] = self.ui.chemChannelInput.text()
        self.parameter['chemical_dilute_MFC_channel'] = self.ui.chemDiluteChannelInput.text()
        self.parameter['humidity_MFC_channel'] = self.ui.humChannelInput.text()
        self.parameter['humidity_dilute_MFC_channel'] = self.ui.humDiluteChannelInput.text()

        # Z_score Parameter
        self.parameter['channels_dict'] = self.ui.zscoreInput.toPlainText()
        
        # Noise Filter Parameter
        self.parameter['process_std'] = self.ui.processSTDInput.text()
        self.parameter['AMS_std'] = self.ui.amsSTDInput.text()
        self.parameter['noise_filter_channel'] = self.ui.filterChannelInput.toPlainText()

        print(self.parameter)


    def run_program(self):

        print('Start process...')

        parameter = self.parameter
        path_list = parameter['path_list']
        self.ui.progressBar.setMaximum(len(path_list))

        for i, path in enumerate(path_list):
            self.ui.progressBar.setValue(i+1)
            sensor_sn = path.split('_')[-2]
            data = pd.read_csv(path)
            run = MainSelectDataProgram(data, parameter, path, sensor_sn)
            run.process()

        print('Process done')



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

