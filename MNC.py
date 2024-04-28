import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers, initializers
from keras.callbacks import EarlyStopping

file_input = r".\97MNC.xlsx"
data_input = pd.read_excel(file_input, sheet_name=0)
feature = ['Lthickness1','HNPP_LNDVI1','HNDVI_LNPP1','Hele_Lndvi2','HTN','Helevation2','Hsilt','HMAP2','HSOC_LTN2','HMAP_LMAT1','elevation', 'MAT', 'MAP', 'NPP', 'SOC', 'TN', 'TP', 'pH','NDVI','AI','silt','thickness','cf','TK','BD','sand','cec','clay']
label = ['MNC']
data_input[feature+label].head()
seed = 0
np.random.seed(seed)
x = data_input[feature].values
y = data_input[label].values
scale_range = (-1, 1)
scaler_feature = preprocessing.MinMaxScaler(scale_range).fit(x)
scaler_label = preprocessing.MinMaxScaler(scale_range).fit(y)
y_scaled = scaler_label.transform(y)
hp_input_dim = x_scaled.shape[1]
hp_epoch = 100000 
hp_batch_size = 120 
hp_aes_opt = optimizers.Adam(lr=0.0007, decay=0.0) 
early_stopping = EarlyStopping(monitor='val_loss', patience=50) 
hp_encoding_zip1 = 1
hp_encoding_zip2 = 2
hp_encoding_dim1 = hp_input_dim-hp_encoding_zip1
hp_encoding_dim2 = hp_input_dim-hp_encoding_zip2
def baseline_aes():
    model = Sequential()
    model.add(Dense(hp_encoding_dim1, input_dim=hp_input_dim, activation='relu'))
    model.add(Dense(hp_encoding_dim2, activation='relu'))
    model.add(Dense(hp_encoding_dim1, activation='relu'))
    model.add(Dense(hp_input_dim, activation='tanh'))
    model.compile(optimizer=hp_aes_opt, loss='mse')
    return model
aes = baseline_aes()
print('Model Summary')
aes.summary()
x_train_aes, x_test_aes = train_test_split(x_scaled, test_size=0.10, random_state=seed)
history_aes = aes.fit(x_train_aes, x_train_aes, batch_size=hp_batch_size, epochs=hp_epoch,
                    verbose=2, validation_data=(x_test_aes,x_test_aes), callbacks=[early_stopping])
fig, ax = plt.subplots()
plt.title('AEs', fontsize=20)
ax.plot(history_aes.history['loss'], c='r', label='Training')
ax.plot(history_aes.history['val_loss'], c='g', label='Validation')
ax.legend(loc='best', shadow=True)
ax.grid(True)

hp_sae_opt = optimizers.Adam(lr=0.0001, decay=0.0)
def baseline_sae():
    model = Sequential()  
    encoder_layer1 = aes.layers[0]
    encoder_layer2 = aes.layers[1]
    model.add(encoder_layer1)
    model.add(encoder_layer2)
    model.add(Dense(128, activation='relu')) 
    model.add(Dense(128, activation='relu')) 
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer=hp_sae_opt, loss='mse')
    return model
print('Model Summary')
baseline_sae().summary()

num_splits = 10
kfold = KFold(n_splits=num_splits, shuffle=True, random_state=seed) 
y_predict = np.zeros((x_scaled.shape[0],1), dtype='float32')
for train_idx, test_idx in kfold.split(x_scaled, y_scaled):
    x_train, y_train = x_scaled[train_idx],  y_scaled[train_idx]
    x_test, y_test = x_scaled[test_idx], y_scaled[test_idx]
    sae = baseline_sae()
    sae.fit(x_train, y_train, batch_size=hp_batch_size, epochs=hp_epoch,
           verbose=2, validation_data=(x_test,y_test), callbacks=[early_stopping])
    y_test_predict = sae.predict(x_test)
    y_test_predict = scaler_label.inverse_transform(y_test_predict)
    y_predict[test_idx] = y_test_predict

lr = LinearRegression()
lr.fit(y, y_predict)
r2 = metrics.r2_score(y, y_predict)
mse = metrics.mean_squared_error(y, y_predict)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y, y_predict)
fig, ax = plt.subplots()
ax.scatter(y, y_predict, c='b', marker='.')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1)
ax.plot(y, lr.predict(y), 'k', lw=1.5)
ax.set_xlabel('Observed MNC (mg g$^-$$^1$)')
ax.set_ylabel('Predicted MNC (mg g$^-$$^1$)')
plt_scale = 0.75
ax.text(y.min(), y.max()*plt_scale, '$y$ = %.2f$x$ + %.2f\n$R^2$ = %.2f\nRMSE = %.2f\nMAE = %.2f' % 
        (lr.coef_, lr.intercept_, r2, rmse, mae))
print("R2 = %.4f\nRMSE = %.4f\nMAE = %.4f\n" % (r2, rmse, mae))
plt.savefig(r'.\Figure-Scatter.tif', dpi=300) 
file_output = r".\MNC_output.xlsx"
excel_writer = pd.ExcelWriter(file_output)
df = pd.DataFrame(y_predict)
df.to_excel(excel_writer, float_format='%.4f', header=['AGB_output'], index=False)
excel_writer.save()

mp = "./model_save.h5"
sae.save(mp)

file_input2 = r".\9986_1km_MNC1.xlsx"
data_input2 = pd.read_excel(file_input2, sheet_name=0)
feature2 = ['HNPP_LNDVI1','Lthickness1','HNDVI_LNPP1','Hele_Lndvi2','HTN','Helevation2','Hsilt','HMAP2','HSOC_LTN2','HMAP_LMAT','elevation', 'MAT', 'MAP', 'NPP', 'SOC', 'TN', 'TP', 'pH','NDVI','AI','silt','thickness','cf','TK','BD','sand','cec','clay']
data_input2[feature2].head()
x_test2 = data_input2[feature2].values
scale_range = (-1, 1)
scaler_feature2 = preprocessing.MinMaxScaler(scale_range).fit(x_test2)
x_test_scaled2 = scaler_feature2.transform(x_test2)
y_test_predict2 = sae.predict(x_test_scaled2)
y_test_predict2 = scaler_label.inverse_transform(y_test_predict2)


file_output2 = r".\9986MNC1.xlsx"
excel_writer = pd.ExcelWriter(file_output2)
df2 = pd.DataFrame(y_test_predict2)
df2.to_excel(excel_writer, float_format='%.4f', header=['MNC'], index=False)
excel_writer.save()

file_input2 = r".\172+9986RCP.xlsx"
data_input2 = pd.read_excel(file_input2, sheet_name=0)
feature2 = ['HNPP&LNDVI4.5','Lthickness4.5','HNDVI&LNPP4.5','Hele&LNDVI4.5','HTN4.5','Hele4.5','Hsilt4.5','HMAP4.5','HSOC&LTN4.5','HMAP&LMAT4.5','elevation', 'MAT4.5(0.26)', 'MAP4.5(11.2)', 'NPP', 'SOC', 'TN', 'TP', 'pH','NDVI','AI','silt','thickness','cf','TK','BD','sand','cec','clay']
data_input2[feature2].head()
x_test2 = data_input2[feature2].values
scale_range = (-1, 1)
scaler_feature2 = preprocessing.MinMaxScaler(scale_range).fit(x_test2) 
x_test_scaled2 = scaler_feature2.transform(x_test2)
y_test_predict2 = sae.predict(x_test_scaled2)
y_test_predict2 = scaler_label.inverse_transform(y_test_predict2)

file_output2 = r".\RCP4.5.xlsx"
excel_writer = pd.ExcelWriter(file_output2)
df2 = pd.DataFrame(y_test_predict2)
df2.to_excel(excel_writer, float_format='%.4f', header=['MNC'], index=False)
excel_writer.save()
file_input2 = r".\172+9986RCP_1.xlsx"
data_input2 = pd.read_excel(file_input2, sheet_name=0)
feature2 = ['HNPP&LNDVI45','Lthickness4.5','HNDVI&LNPP45','Hele&LNDVI45','HTN45','Hele4.5','Hsilt4.5','HMAP45','HSOC&LTN45','HMAP&LMAT45','elevation', 'MAT45', 'MAP45', 'NPP', 'SOC', 'TN', 'TP', 'pH','NDVI','AI','silt','thickness','cf','TK','BD','sand','cec','clay']
data_input2[feature2].head()
x_test2 = data_input2[feature2].values
scale_range = (-1, 1)
scaler_feature2 = preprocessing.MinMaxScaler(scale_range).fit(x_test2)
x_test_scaled2 = scaler_feature2.transform(x_test2)
y_test_predict2 = sae.predict(x_test_scaled2)
y_test_predict2 = scaler_label.inverse_transform(y_test_predict2)
file_output2 = r".\RCP45.xlsx"
excel_writer = pd.ExcelWriter(file_output2)
df2 = pd.DataFrame(y_test_predict2)
df2.to_excel(excel_writer, float_format='%.4f', header=['MNC'], index=False)
excel_writer.save()


file_input2 = r".\172+9986RCP_1.xlsx"
data_input2 = pd.read_excel(file_input2, sheet_name=0)
feature2 = ['HNPP&LNDVI85','Lthickness8.5','HNDVI&LNPP85','Hele&LNDVI85','HTN85','Hele8.5','Hsilt8.5','HMAP85','HSOC&LTN45','HMAP&LMAT85','elevation', 'MAT85', 'MAP85', 'NPP', 'SOC', 'TN', 'TP', 'pH','NDVI','AI','silt','thickness','cf','TK','BD','sand','cec','clay']

data_input2[feature2].head()
x_test2 = data_input2[feature2].values
scale_range = (-1, 1)
scaler_feature2 = preprocessing.MinMaxScaler(scale_range).fit(x_test2)
x_test_scaled2 = scaler_feature2.transform(x_test2)
y_test_predict2 = sae.predict(x_test_scaled2)
y_test_predict2 = scaler_label.inverse_transform(y_test_predict2)

file_output2 = r".\RCP85.xlsx"
excel_writer = pd.ExcelWriter(file_output2)
df2 = pd.DataFrame(y_test_predict2)
df2.to_excel(excel_writer, float_format='%.4f', header=['MNC'], index=False)
excel_writer.save()


file_input2 = r".\25MNC.xlsx"
data_input2 = pd.read_excel(file_input2, sheet_name=0)
feature2 = ['HNPP_LNDVI','Lthickness','HNDVI_LNPP','Hele_Lndvi','HTN','Helevation','Hsilt','HMAP','HSOC_LTN','HMAP_LMAT','elevation', 'MAT', 'MAP', 'NPP', 'SOC', 'TN', 'TP', 'pH','NDVI','AI','silt','thickness','cf','TK','BD','sand','cec','clay']
data_input2[feature2].head()
x_test2 = data_input2[feature2].values

scale_range = (-1, 1)
scaler_feature2 = preprocessing.MinMaxScaler(scale_range).fit(x_test2) 

x_test_scaled2 = scaler_feature2.transform(x_test2)
y_test_predict2 = sae.predict(x_test_scaled2)
y_test_predict2 = scaler_label.inverse_transform(y_test_predict2) 


file_output2 = r".\25valid.xlsx"
excel_writer = pd.ExcelWriter(file_output2)
df2 = pd.DataFrame(y_test_predict2)
df2.to_excel(excel_writer, float_format='%.4f', header=['MNC'], index=False)
excel_writer.save()






