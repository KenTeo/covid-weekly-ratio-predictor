# Real Time covid cases prediction

This Project is based on using time-series modelling methods to predict the covid infection rate in Singapore.
The machine learning can potentially help us to predict the weekly community infection ratio 7 days in advance. 
This can help the healthy ministry to decide if there is a need to change covid restriction measures for the country.
For example, if the weekly community infection ratio is predicted to exceed 1.0 in 7 days, the government can encourage
people to work from home. The data is shown using a interactive dashboard which allows users to have a means to forcast 
the covid situation in Singapore in the coming weeks.

**The project is hosted on a free website on herokuapp. See link below:**
https://covidsit-singapore-prediction.herokuapp.com/

### Data Extraction:

Data is copied from Singapore MOH API. Then, the relevant data is extracted from the JSON file copied.
The Path to extract data from the website in realtime is shown below:

*The Covid data are obtain from the following API:*
https://covidsitrep.moh.gov.sg/_dash-layout

    API_path = 'https://covidsitrep.moh.gov.sg/_dash-layout'
    MOH = requests.get(API_path).json()
    date = MOH['props']['children'][1]['props']['children'][2]['props']['children'][0]['props']['figure']['data'][1]['x']
    comm_cases = MOH['props']['children'][1]['props']['children'][2]['props']['children'][0]['props']['figure']['data'][1]['y']
    dorm_cases = MOH['props']['children'][1]['props']['children'][2]['props']['children'][0]['props']['figure']['data'][3]['y']
    import_cases = MOH['props']['children'][1]['props']['children'][2]['props']['children'][0]['props']['figure']['data'][5]['y']

### Feature Engineering
The weekly infection cases and the weekly infection ratio as to be post-process in-house as the data are not available in the API.
**def weekly_cases**  -  to calculate the total cases in one week
**def weekly_ratio**  -  To calculate the week-to-week infection ratio

*For more details on week-to-week infection ratio, check out the link below:*
https://www.todayonline.com/singapore/explainer-what-weekly-infection-growth-rate-and-how-does-it-help-us-assess-covid-19

    def weekly_cases(select, column_name):
        cases = [0, 0, 0, 0, 0, 0]
        for x in range(6, len(select)):
            weekly_avg = (select.loc[x, column_name] +
                          select.loc[x-1, column_name] +
                          select.loc[x-2, column_name] +
                          select.loc[x-3, column_name] +
                          select.loc[x-4, column_name] +
                          select.loc[x-5, column_name] +
                          select.loc[x-6, column_name])
            cases.append(weekly_avg)
        return cases

    def weekly_ratio(select, column_name):
        ratio = [0.0]*13
        for x in range(13, len(select)):
            if select.loc[x-7, column_name] == 0:
                ratio.append(ratio[-1])
            else:
                avg_ratio = (select.loc[x, column_name])/select.loc[x-7, column_name]
                ratio.append(avg_ratio)
        return ratio

### LSTM prediction model
LSTM model has been used to predict the weekly infection ratio as it has been shown to be more robust in predicting than ARIMA model.
The input is the following 6 features for the past 14 days; comm_weekly_cases, comm_weekly_ratio, dorm_weekly_cases, dorm_weekly_ratio, import_weekly_cases, import_weekly_ratio. While the output is the 6 features in future by 1 day.
**See below for the summary of the LSTM model:**

    Model: "sequential_21"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm_37 (LSTM)              (None, 14, 64)            18176     
     lstm_38 (LSTM)              (None, 32)                12416     
     dropout_17 (Dropout)        (None, 32)                0         
     dense_17 (Dense)            (None, 6)                 198       
                                                                     
    =================================================================
    Total params: 30,790
    Trainable params: 30,790
    Non-trainable params: 0

## Website
### Data summary
At the top of the webpage, the most updated day, cases on most updated day and total weekly cases on most updated day will be displayed to inform users of the website. Both the cases and ratio for the day and week will be shown.
**As shown in the image below:**
![image](https://drive.google.com/uc?export=view&id=1fGrvysOj4n7ilOCAxVkoYkVLQREeus13)

### Line plot for covid weekly infection ratio
Chartjs is used to plot the lineplot of the week-to-week infection ratio for community, dormitory and imported. Chartjs is a interactive graph that allows users to view the value when the mouse is over the point and clicking the legend of the specific variable can hide or show the variable as the user wishes.
**The graph is as shown in the image below:**
![image](https://drive.google.com/uc?export=view&id=1_ZlT1Hscpv4rGFnGgIeP5aCkii6yUi8w)
**When the mouse is over a point. (value is shown, see below)**
![image](https://drive.google.com/uc?export=view&id=15IX5KK0azm_5noU1tC5MzI2WtVhWOEsK)
The prediction of dormitory and imported is more volatile because the covid restrictions are more extreme which has cause the number of infection
cases in dormitory and imported cases to change drastically as a result of different covid restriction measures for international travel and dormitory.
Hence, as covid restriction has been less drastic and the number of cases has been more stable. The predicted model for the community weekly ratio is
quite stable and can predict the number of cases relatively well. 

### Overview
The overview of the webpage is shown below: 
![image](https://drive.google.com/uc?export=view&id=1PlMaQsQpc60087Gv9rzLz8YMb2mpJBfQ)
(Link: https://covidsit-singapore-prediction.herokuapp.com/)
*Click Covidsitrep and Data.gov to enter the covid data sources.*
