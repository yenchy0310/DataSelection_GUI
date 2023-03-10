import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import plotly package
# import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
# import plotly.offline as offline
# from plotly.offline import init_notebook_mode
# import cufflinks as cf
# cf.go_offline(connected=True)
# init_notebook_mode(connected=True)


def savePathX(folder_name):
    savePath = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    return savePath    

class model:
    '''channel: analysis channel (ex: 440nm #1_mv_comp)
       x_train: training data for model building
       y_train: signal of ppm
       x_test: testing data for model evaluation
       y_test: signal of ppm       
       '''
    
    def __init__(self, folder_name, sensor_number, channel, x_train, y_train, x_test, y_test, model_name, degree, ppm, output_modify=1, shift=0, multiple=1, humidity_feature=True):
        self.folder_name = folder_name
        self.sensor_number = sensor_number
        self.channel = channel
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_name = model_name
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.output_modify = output_modify
        self.shift = shift
        self.multiple = multiple
        self.ppm = ppm
        self.humidity_feature = humidity_feature
        
          
    
    def regression(self):
        
        if self.humidity_feature == True:
            x = self.x_train
            X_ = self.x_test
        else:
            x = self.x_train[['Temperature']]
            X_ = self.x_test[['Temperature']]
      
        x = self.poly.fit_transform(x)
        X_ = self.poly.fit_transform(X_)

        # training set 
        model = self.model_name.fit(x, self.y_train)
        yfit = model.predict(x)
        self.coeff = model.coef_
        self.intercept = model.intercept_
        self.rmse_train = np.sqrt(mean_squared_error(self.y_train, yfit))
        
        # testing set
        self.y_pred = model.predict(X_) * self.output_modify
        self.rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        
        # Temperature and humidity boundary
        self.T_lower_bound = -50 #???????????????
        self.T_upper_bound = 90 #???????????????
        self.H_lower_bound = 0 #???????????????
        self.H_upper_bound = 100 #???????????????
        
#         self.T_lower_bound = np.floor(self.x_train['Temperature'].min()) #???????????????
#         self.T_upper_bound = np.ceil(self.x_train['Temperature'].max()) #???????????????
#         self.H_lower_bound = np.floor(self.x_train['Humidity'].min()) #???????????????
#         self.H_upper_bound = np.ceil(self.x_train['Humidity'].max()) #???????????????

        return self.intercept, self.coeff

    
    
        
    def save_sensor_side_coef(self):

        # create a folder to store csv file

        self.savePath = savePathX(self.folder_name)
        '''self.savePath = os.path.join(os.getcwd(), self.folder_name)
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)'''

                
        # coefficient???0???
        if (self.humidity_feature == False)&(self.degree==1):
            self.coeff = np.append(self.coeff, [0] * self.degree)

        elif (self.humidity_feature == False)&(self.degree==2):
            self.coeff = np.insert(self.coeff, 1, 0)
            self.coeff = np.append(self.coeff, [0] * self.degree)

        elif (self.humidity_feature == False)&(self.degree==3):
            self.coeff = np.insert(self.coeff, 1, 0)
            self.coeff = np.insert(self.coeff, 3, 0)
            self.coeff = np.insert(self.coeff, 4, 0)
            self.coeff = np.append(self.coeff, [0] * self.degree)
                         
            
        coef_list = [self.T_lower_bound, self.T_upper_bound, self.H_lower_bound, self.H_upper_bound, self.degree, self.ppm, 1, 0, self.intercept] # ???1,0?????????FW tool??????????????????????????????????????????1?????????0
        
        for i in self.coeff:
            coef_list.append(i)
        
        df_coef = pd.DataFrame()
        df_coef['coefficient'] = coef_list #model coefficient 
        coefAmount = len(self.coeff)
        coef = ['coef'] * coefAmount        
        header = ['T low', 'T high', 'H low', 'H high', 'degree', 'ppm', 'multiple', 'shift', 'intercept'] + coef
                        
        df_coef.T.to_csv(self.savePath + '\{}({}ppm)(T={}~{})(H={}~{})(degree={})({})(shift{})(multiple{}).csv'.format(self.sensor_number, self.ppm, self.T_lower_bound, self.T_upper_bound, self.H_lower_bound, self.H_upper_bound, self.degree, self.channel, self.shift, self.multiple), header=header, index=False) 
        
        return coef_list[8:]
              

    def save_white_card_side_coef(self, parameter=1):

        # create a folder to store csv file

#         savePath = model.__create_savePath()
        self.savePath = savePathX(self.folder_name)
        '''
        self.savePath = os.path.join(os.getcwd(), self.folder_name)
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        '''
                 
        coef_list = [self.T_lower_bound, self.T_upper_bound, self.H_lower_bound, self.H_upper_bound, self.degree, -23, 1, 0] # -23=ppm, 1=multiple, 0=shift ?????????????????????phage regression????????????
        
        coef_list.append(self.intercept*parameter)
        for i in self.coeff:
            coef_list.append(i*parameter)
        
        df_coef = pd.DataFrame()        
        df_coef['coefficient'] = coef_list  #model coefficient
        coefAmount = len(self.coeff)
        coef = ['coef'] * coefAmount        
        header = ['T low', 'T high', 'H low', 'H high', 'degree', 'ppm', 'multiple', 'shift', 'intercept'] + coef
        
        df_coef.T.to_csv(self.savePath + '\{}(T={}~{})(H={}~{})(degree={})({})(parameter={:.4f}).csv'.format(self.sensor_number, self.T_lower_bound, self.T_upper_bound, self.H_lower_bound, self.H_upper_bound, self.degree, self.channel, parameter), header=header, index=False)

        

    def loss(self):
        print('RMSE_train= {:.2f}'.format(self.rmse_train))
        print('RMSE_test= {:.2f}'.format(self.rmse_test))
       
    
    '''Ignore RH30% prediction performance'''
    # def loss_noRH30(self):
    #     y_pred_noRH30 = self.y_pred[self.each_step_data_point:]
    #     y_test_noRH30 = self.y_test[self.each_step_data_point:]
    #     self.rmse_noRH30 = np.sqrt(mean_squared_error(y_pred_noRH30, y_test_noRH30))      
        
        
    def coef(self):
        print(self.poly.get_feature_names())
        coefficient = np.insert(self.coeff, 0, self.intercept)
        print('({}~{})({}~{})'.format(self.T_lower_bound, self.T_upper_bound, self.H_lower_bound, self.H_upper_bound), coefficient)
        return self.intercept, self.coeff
                
        
    def plot(self):  
        filename = '{} (T={}~{}) (H={}~{}) (degree={}) ({}) (shift{}_ppmx{}) (RMSE y_pred_scale= {:.2f})'.format(self.ppm, self.T_lower_bound, self.T_upper_bound, self.H_lower_bound, self.H_upper_bound, self.degree, self.channel, self.shift, self.multiple, self.rmse_test ) 
        self.df_result = pd.DataFrame()
        self.df_result['y_true'] = self.y_test
        self.df_result['y_pred'] = self.y_pred
        self.df_result['Humidity'] = self.x_test['Humidity']
        self.df_result['Temperature'] = self.x_test['Temperature']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.df_result['y_true'], name='y_true'))
        fig.add_trace(go.Scatter(y=self.df_result['y_pred'], name='y_true'))
        fig.update_layout(title=filename)
        
#         self.df_result.iplot(y=['y_true', 'y_pred'], 
#                         kind='scatter', 
#                         yTitle='ppm', 
#                         title=filename
#                             )
        fig.show()

    def save_plot(self): 
        # save prediction result 
#         savePath = model.__create_savePath()
        filename = '{} (T={}~{}) (H={}~{}) (degree={}) ({}) (shift{}_ppmx{}) (RMSE y_pred_scale= {:.2f})'.format(self.ppm, self.T_lower_bound, self.T_upper_bound, self.H_lower_bound, self.H_upper_bound,  self.degree, self.channel, self.shift, self.multiple, self.rmse_test )       
        plt.plot(self.df_result['y_true'])
        plt.plot(self.df_result['y_pred'])
        plt.plot(self.df_result['Temperature'])
        plt.ylabel('ppm')
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),  loc=3, ncol=3, mode="expand")
        plt.title(filename, y=1.15, fontsize=10)
#         plt.savefig(savePath + '/' + filename + '.png', dpi=200, bbox_inches='tight')
        plt.savefig(self.savePath + '/' + filename + '.png', dpi=200, bbox_inches='tight')
        plt.close() 
        # plt.show()
        
    
