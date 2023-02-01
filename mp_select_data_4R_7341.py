import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
from collections import namedtuple

class MainSelectDataProgram():
    def __init__(self, data, parameter, path, sensor_sn):
        self.data = data
        self.parameter = parameter
        self.data_path = path
        self.sensor_sn = sensor_sn
        self.save_path = self.save_folder_path()

        self.number = parameter['number']
        self.calibration = parameter['calibration']
        self.QC = parameter['QC']
        self.whether_select_data = parameter['whether_select_data']
        self.plot_channel = parameter['analysis_channel']

    def process(self):

        kalman_filter = OneDimensionFilter(self.data, self.parameter)
        self.data = kalman_filter.process()
        
        data_preprocess = DataPreprocess(self.data, self.parameter)
        self.data = data_preprocess.process()

        select_range = SelectRangeOfEachTemperatureData(self.data, self.parameter)
        temperature_data_select_range = select_range.create_select_range()
        print('temperature_data_select_range', temperature_data_select_range)

        select_data_7341 = SelectData7341(self.data, data_path=self.data_path)

        if len(temperature_data_select_range) == 0:
            save_name = 'select_entire_data_QC{}'.format(self.number)
            select_data_7341.save_entire_data(save_path=self.save_path, save_name=save_name)
            print('Save entire data done')

        all_select_range = {}
        for temperature, select_range_dict in temperature_data_select_range.items():

            if self.calibration == True and self.whether_select_data == True:
                save_name = 'select_data_calibration_{}C'.format(temperature)
                select_data_7341.select_data(select_range=select_range_dict)
                select_data_7341.save_select_data(save_path=self.save_path, save_name=save_name)
                print('Save select data done')
                all_select_range.update(select_range_dict)
            
        
            elif self.calibration == True and self.whether_select_data == False:
                save_name = 'select_entire_data_calibration_{}C'.format(temperature)
                select_data_7341.save_entire_data_by_temp(save_path=self.save_path, save_name=save_name, temperature=temperature)
                print('Save entire data done')

            elif self.QC == True and self.whether_select_data == True:
                save_name = 'select_data_blind_test_{}C_{}'.format(temperature, self.number)
                select_data_7341.select_data(select_range=select_range_dict)
                select_data_7341.save_select_data(save_path=self.save_path, save_name=save_name)
                print('Save select data done')
                all_select_range.update(select_range_dict)

            elif self.QC == True and self.whether_select_data == False:
    #             save_name = 'select_entire_data_blind_test_{}C_{}'.format(temperature, number)
    #             select_data_7341.save_entire_data_by_temp(save_path=save_path, save_name=save_name, temperature=temperature)
                save_name = 'select_entire_data_QC{}'.format(self.number)
                select_data_7341.save_entire_data(save_path=self.save_path, save_name=save_name)

                print('Save entire data done')
                
            else:
                print('Choose incorrect')

        
        if self.calibration == True:
            save_name = 'calibration'

        elif self.QC == True:
            save_name = 'QC_{}'.format(self.number)
        
        draw_figure = DrawFigure(self.data, self.save_path, save_name, all_select_range, self.sensor_sn, self.plot_channel)
        draw_figure.plot_data()
        draw_figure.save_figure()

    def save_folder_path(self):

        save_path = os.path.join(os.path.split(os.path.split(self.data_path)[0])[0], self.sensor_sn)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        return save_path

class DataPreprocess():
    def __init__(self, data, parameter
                #  chemical_MFC_max_flow_rate=0.8,
                #  chemical_dilute_MFC_max_flow_rate=0.8,
                #  humidity_MFC_max_flow_rate=4,
                #  humidity_dilute_MFC_max_flow_rate=4,
                #  cylinder_concentration=1000, 
                #  chemical_MFC_channel='MFC#1',
                #  chemical_dilute_MFC_channel='MFC#2',
                #  humidity_MFC_channel='MFC#3',
                #  humidity_dilute_MFC_channel='MFC#4',
                #  analysis_channel='583nm #1',
                #  remove_spike = True,
                #  remove_spike_threshold=0.1,
                #  clear_compensation=False
                 ):

        self.data = data
        self.chemical_MFC_max_flow_rate = float(parameter['chemical_MFC_max_flow_rate'])
        self.chemical_dilute_MFC_max_flow_rate = float(parameter['chemical_dilute_MFC_max_flow_rate'])
        self.humidity_MFC_max_flow_rate = float(parameter['humidity_MFC_max_flow_rate'])
        self.humidity_dilute_MFC_max_flow_rate = float(parameter['humidity_dilute_MFC_max_flow_rate'])
        self.chemical_MFC_channel = parameter['chemical_MFC_channel']
        self.chemical_dilute_MFC_channel = parameter['chemical_dilute_MFC_channel']
        self.humidity_MFC_channel = parameter['humidity_MFC_channel']
        self.humidity_dilute_MFC_channel = parameter['humidity_dilute_MFC_channel']
        self.cylinder_concentration = int(parameter['cylinder_concentration'])
        self.analysis_channel = parameter['analysis_channel']
        self.remove_spike = parameter['remove_spike']
        self.remove_spike_threshold = float(parameter['remove_spike_threshold'])
        self.channels_dict = eval(parameter['channels_dict']) # string 轉 dict

        # self.clear_compensation = clear_compensation

    def process(self):

        self.change_column_name()
        self.data = self.add_ppm_column()
        
        if self.remove_spike == True:
            self.data = self.remove_spike_function()

        self.data = self.Z_score_transform()    
        # if self.clear_compensation == True:
        #     self.clear_channel_compensation()

        return self.data

    def change_column_name(self):
    # Change columns name
        self.data.rename(columns={'External Temp':'Temperature', 'External Humidity':'Humidity'}, inplace=True)
    
    def add_ppm_column(self):
        '''Adding ppm column by converting MFC1 value'''
        total_flow_rate = (self.data[self.chemical_MFC_channel]*self.chemical_MFC_max_flow_rate) + (self.data[self.chemical_dilute_MFC_channel]*self.chemical_dilute_MFC_max_flow_rate) + (self.data[self.humidity_MFC_channel]*self.humidity_MFC_max_flow_rate) + (self.data[self.humidity_dilute_MFC_channel]*self.humidity_dilute_MFC_max_flow_rate)
        
        self.data['total flow rate'] = total_flow_rate
       
        if self.data['total flow rate'].sum() == 0:
             self.data['ppm'] = [0]*self.data.shape[0]
        else:
             self.data['ppm'] = self.data[self.chemical_MFC_channel]*self.chemical_MFC_max_flow_rate/total_flow_rate*self.cylinder_concentration
        
        return self.data 

    def remove_spike_function(self):

        print('remove_spike_threshold =', self.remove_spike_threshold)
        AMS1_spike_index = self.data[abs(self.data[self.analysis_channel]-self.data[self.analysis_channel][0:10].mean())/self.data[self.analysis_channel][0:10].mean()> self.remove_spike_threshold].index
        AMS2_spike_index = self.data[abs(self.data[self.analysis_channel]-self.data[self.analysis_channel][0:10].mean())/self.data[self.analysis_channel][0:10].mean()> self.remove_spike_threshold].index
        print('AMS1_spike_index', AMS1_spike_index)
        print('AMS2_spike_index', AMS2_spike_index)
        spike_index = np.append(AMS1_spike_index, AMS2_spike_index)
        self.data.drop(labels=spike_index, inplace=True)
        self.data.reset_index(inplace=True)

        return self.data 

    def Z_score_transform(self):
    
        """Z_score transform function
        df_data: dataframe
        channels: list(str), the channels use to calculate average and standard deviation
        main_channels: str, the channel uses to calculate z_score"""

        for _, parameters in self.channels_dict.items():
            channels = parameters[0]
            Z_score_main_channel = parameters[1]
            mean = self.data.loc[:,channels].mean(axis=1)
            std = self.data.loc[:,channels].std(axis=1, ddof=0)
        
            if 'Filtered' in Z_score_main_channel:
                prefixed = Z_score_main_channel.split('_')[0]
                tailed = Z_score_main_channel.split('_')[1]
                self.data['{}_Z_score_{}'.format(prefixed, tailed)] = (self.data[Z_score_main_channel] - mean) / std
            else:    
                self.data['Z_score_{}'.format(Z_score_main_channel)] = (self.data[Z_score_main_channel] - mean) / std
                
            print("Z_score channel", channels)
            print("Z_score_main_channel", Z_score_main_channel)

        return self.data 
                     
class OneDimensionFilter():

    def __init__(self, data, parameter
                #  channel=['470nm #1', '583nm #1',
                #           '510nm #1', '550nm #1', 
                #           '620nm #1', '670nm #1',
                #           '470nm #2', '583nm #2',
                #           '510nm #2', '550nm #2', 
                #           '620nm #2', '670nm #2',],
                # process_std = 1.2,
                # AMS_std = 1.8
                          ):
        self.data = data
        self.channel = eval(parameter['noise_filter_channel'])
        self.process_std = float(parameter['process_std'])
        self.AMS_std = float(parameter['AMS_std'])
        self.gaussian = namedtuple('Gaussian', ['mean', 'var'])
        self.gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])

        print('noise_filter_channel', self.channel)

    def process(self):

        for wavelength in self.channel:
            filter_data = []
            process_model = self.gaussian(0., self.process_std)
            x = self.gaussian(10000, 3000) # Initial state guess

            for signal in self.data[wavelength]:
                prior = self.predict(x, process_model)
                x = self.update(prior, self.gaussian(signal, self.AMS_std))
                filter_data.append(x.mean)
            self.data[f'Filtered_{wavelength}'] = filter_data
            
        print("Kalman filter channel", self.channel)
        
        return self.data
    
    def predict(self, initial_state, process_model):
        return self.gaussian(initial_state.mean + process_model.mean, initial_state.var + process_model.var)

    def update(self, prior, likelihood):
        posterior = self.gaussian_multiply(likelihood, prior)
        return posterior

    def gaussian_multiply(self, pdf1, pdf2):  ## pdf: probability distribution function
        mean = (pdf1.var * pdf2.mean + pdf2.var * pdf1.mean)/(pdf1.var + pdf2.var)
        variance = (pdf1.var * pdf2.var)/(pdf1.var + pdf2.var)
        return self.gaussian(mean, variance)

class SelectRangeOfEachTemperatureData():
    def __init__(self, data, parameter):
        
        self.data = data
        self.datapoint_number = int(parameter['datapoint_number'])
        self.buffer = int(parameter['buffer'])
        
    def create_select_range(self):
        """找出每個溫度對應的select range dictionary"""
        
        temp_data_dict = self.get_each_temperature_data()
    
        # 找出每個溫度需要擷取資料的範圍
        # temp_select_index_dict = {key:temperature, value:select_range_dict}
        temp_select_range_dict = {}
        for temp, temp_data in temp_data_dict.items():
            select_range = self.auto_select(temp_data)

            # dict_set為list，elements為select資料的的index
            # 把list轉換成dictionary的型態
            select_range_dict = {}
            for i, index in enumerate(select_range):
                select_range_dict[f'{temp}_{i}'] = (index, self.datapoint_number)
                # temp_select_range_dict[f'{temp}_{i}'] = (index, self.datapoint_number)
                
            if len(select_range_dict) != 0:
                temp_select_range_dict[temp] = select_range_dict

            else:
                continue

        return temp_select_range_dict  

    def get_each_temperature_data(self):
        "擷取資料內各溫度對應的資料"
               
        # 取出溫度階 
        self.data['temp_steps'] = self.data['Temperature'].apply(lambda x: int(round(x/100, 1)*100) if abs(x) >= 10 else int(round(x, 0)))   
        print('temp_steps', self.data['temp_steps'].unique())

        # 創造溫度對dataset的dictionary資料結構
        # temp_data_dict = {key:temperature, value: the temperature dataframe}
        temp_data_dict = {}
        for temp in self.data['temp_steps'].unique():
            temp_index = self.data[self.data['temp_steps'] == temp].index 
            df_temp = self.data.iloc[temp_index, :]
            df_temp.index = temp_index 
            temp_data_dict[temp] = df_temp
        
        return temp_data_dict        
    
    def auto_select(self, data):
        """data: dataframe
           total_flow_rate: int
           MFC1_max_flow_rate: int
           cylinder_concentration: int"""

        flag = 0
        dict_set = []
        
        for index in data.index:
            ppm = data.loc[index,'ppm']

            if flag == 0: #取0ppm
#                 if ppm>3.8 and ppm<4.2:
#                     dict_set.append(index-self.buffer)    
#                     flag=4
                
#                 elif ppm>7.2 and ppm<7.7:
#                     dict_set.append(index-self.buffer)    
#                     flag=7.5    

                if ppm>9.8 and ppm<10.2:
                    dict_set.append(index-self.buffer)    
                    flag=10

#                 elif ppm>24.8 and ppm<25.2:
    #                 dict_set.append(index-self.buffer)    
#                     flag=25
        
                elif ppm>39.8 and ppm<40.2:
#                     dict_set.append(index-self.buffer)    
                    flag=40

#                 elif ppm>44.8 and ppm<45.2:
#                     dict_set.append(index-self.buffer)    
#                     flag=45

                elif ppm>59.8 and ppm<60.2:
#                     dict_set.append(index-self.buffer)     
                    flag=60
                    
                elif ppm>79.8 and ppm<80.2:
                    dict_set.append(index-self.buffer)     
                    flag=80

#             elif flag == 4: #取4ppm
#                 if ppm<2:
#                     dict_set.append(index-self.buffer)    
#                     flag=0

#             elif flag == 7.5: #取7.5ppm
#                 if ppm<2:
#                     dict_set.append(index-self.buffer)    
#                     flag=0

            elif flag == 10: #取10ppm
                if ppm<2:
                    dict_set.append(index-self.buffer)    
                    flag=0
                    
                if ppm>39.8 and ppm<40.2:
                    dict_set.append(index-self.buffer)    
                    flag=40

#             elif flag == 25: #取25ppm
#                 if ppm<2:
#                     dict_set.append(index-self.buffer)    
#                     flag=0
                    
            elif flag == 40: #取40ppm
                if ppm<2:
                    dict_set.append(index-self.buffer)    
                    flag=0
                    
                if ppm>59.8 and ppm<60.2:
                    dict_set.append(index-self.buffer)    
                    flag=60
                    
                if ppm>9.8 and ppm<10.2:
                    dict_set.append(index-self.buffer)    
                    flag=10

#             elif flag == 45: #取45ppm
#                 if ppm<2:
#                     dict_set.append(index-self.buffer)  
#                     flag=0

#                 if ppm<2:
#                     dict_set.append(index-self.buffer)   
#                     flag=0

#                 if ppm>44.8 and ppm<40.2:
#                     dict_set.append(index-self.buffer)    
#                     flag=40

            elif flag == 60: #取60ppm
                if ppm<2:
                    dict_set.append(index-self.buffer)    
                    flag=0
                    
                if ppm>39.8 and ppm<40.2:
                    dict_set.append(index-self.buffer)    
                    flag=40
                    
            elif flag == 80: #取80ppm
                if ppm<2:
                    dict_set.append(index-self.buffer)   
                    flag=0
                    
                if ppm>59.8 and ppm<60.2:
#                     dict_set.append(index-self.buffer)    
                    flag=60
                    
        # print('__auto_select', dict_set)
        return dict_set
    
class SelectData7341():
    def __init__(self, data, data_path):

        self.data = data
        self.data_path = data_path
    
    def select_data(self, select_range):        
        '''selected range format

           Set datapoint in the dictionary
           selected_range = {'region 1': (end, datapoint),
                             'region 2': (900, 10), 
                             'region 3': (1200, 10),
                             'region 4': (1500, 10),}'''

        # selected data
        df_list=[]    
        for b, datapoint in select_range.values():
            select_data = self.data[b-datapoint:b]
            df_list.append(select_data)
                
        # concate select data
        self.df_select_data = pd.concat(df_list).reset_index(drop=True)
        
    def save_select_data(self, save_path, save_name):
        self.df_select_data.to_csv(save_path + '/'+ save_name + '.csv', index=False)

    def save_entire_data_by_temp(self, save_path, save_name, temperature):
        self.data[self.data['temp_steps']==temperature].to_csv(save_path + '/'+ save_name + '.csv', index=False)
        
    def save_entire_data(self, save_path, save_name):
        self.data.to_csv(save_path + '/'+ save_name + '.csv', index=False)   
    
class DrawFigure():
    def __init__(self, data, save_path, save_name, select_range={}, sensor_sn=None, plot_channel='583nm #1'): 
        self.data = data
        self.save_path = save_path
        self.save_name = save_name
        self.select_range = select_range
        self.sensor_sn = sensor_sn
        self.plot_channel = plot_channel

    def plot_data(self):
                
        '''selected range format
           selected_range = {'region 1': (b-datapoint, b),
                             'region 2': (1200-datapoint, 1200)}'''
        
        mapping_channel = [None]*self.data.shape[0]
        mapping_ppm = [None]*self.data.shape[0]
        for b, datapoint in self.select_range.values():
            mapping_channel[b-datapoint:b] = self.data[self.plot_channel][b-datapoint:b]
            mapping_ppm[b-datapoint:b] = self.data['ppm'][b-datapoint:b]

        
        self.fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8,8), constrained_layout=True)
        # self.fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
        self.fig.suptitle(self.sensor_sn, fontsize=14 , x=0.1, y=1.02)
        axs[0].set_title('Humidity')
        axs[0].plot(self.data['Humidity'], color='blue', label='Humidity')
        axs[1].set_title('Temperature')
        axs[1].plot(self.data['Temperature'], color='red', label='Temperature')
        axs[2].set_title(self.plot_channel)
        axs[2].plot(self.data[self.plot_channel], color='#00AA55', label=self.plot_channel)
        axs[2].plot(mapping_channel, color='#CC00CC', label='mapping_channel')
        axs[3].plot(self.data['ppm'], color='orange', label='ppm')
        axs[3].plot(mapping_ppm, color='#00BBFF', label='mapping_ppm')
        

        
    def save_figure(self):
        self.fig.savefig(self.save_path + '/' + self.save_name + '.jpeg', bbox_inches='tight',dpi=100)

    # def plot_data(self):
                
    #     '''selected range format
    #        selected_range = {'region 1': (b-datapoint, b),
    #                          'region 2': (1200-datapoint, 1200)}'''
        
    #     self.fig = make_subplots(rows=3, 
    #                              cols=1, 
    #                              shared_xaxes=True,
    #                              vertical_spacing=0.1,
    #                              subplot_titles=["Temperature / Humidity", self.plot_channel],
    #                              row_heights=[30,30,30],
    #                              specs=[[{"secondary_y": True}],[{"secondary_y": False}],[{"secondary_y": False}]]
    #                              )

    #     self.fig.add_trace(go.Scatter(y=self.data['Humidity'], name='Humidity'), row=1, col=1, secondary_y=True)                         
    #     self.fig.add_trace(go.Scatter(y=self.data['Temperature'], name='Temperature'), row=1, col=1, secondary_y=False)
    #     self.fig.update_yaxes(title_text='Humidity', row=1, col=1, secondary_y=True)
    #     self.fig.update_yaxes(title_text='Temperature', row=1, col=1, secondary_y=False)

    #     self.fig.add_trace(go.Scatter(y=self.data[self.plot_channel], name=self.plot_channel), row=2, col=1)
            
    #     mapping_channel = [None] * self.data.shape[0]
    #     mapping_ppm = [None] * self.data.shape[0]

    #     for b, datapoint in self.select_range.values():
    #         mapping_channel[b-datapoint:b] = self.data[self.plot_channel][b-datapoint:b]
    #         mapping_ppm[b-datapoint:b] = self.data['ppm'][b-datapoint:b]

    #     self.fig.add_trace(go.Scatter(y=mapping_channel, name='mapping_channel'), row=2, col=1)
    #     self.fig.add_trace(go.Scatter(y=self.data['ppm'], name='ppm'), row=3, col=1)
    #     self.fig.add_trace(go.Scatter(y=mapping_ppm, name='mapping_ppm'), row=3, col=1)  
    #     self.fig.update_layout(title=self.sensor_sn, width=900, height=800)

    #     self.fig.show()

    # def save_figure(self):
    #     self.fig.write_image(self.save_path + '/' + self.save_name + '.jpeg')
       






        