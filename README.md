# NCAA Score Predictions

## Objective:

Predict scores of Division I Men's NCAA matchups based on historical performance metrics.

## Featured Techniques:

- Web Scraping 
- Feature Engineering 
- Supervised Learning
- Linear Regression (OLS)

## Methodology:

![model_design](https://i.loli.net/2020/06/17/3e9YkJ7guAHFsUb.png)



## Analysis:

## Results:

![act vs pred points](https://i.loli.net/2020/06/17/ivatoHVCR5Qyzuw.png)

## ![act vs pred margin and total](https://i.loli.net/2020/06/17/7zTZY8RkBLAm6ep.png)

## Future Work:

Ongoing work on this project will entail training a single (simpler) model that attempts to predict the *score differential* for a given game in lieu of independently predicting the score of each team.  

![future_model](https://i.loli.net/2020/06/17/kCR6MTfdmFejvlb.png)

Score differential will be determined as the home team's score minus the away team's score, such that positive score differentials indicate a victorious home team and negative differentials indicate the away team victories. 

## Data Sources:

https://www.sports-reference.com/cbb/schools/

https://console.cloud.google.com/marketplace/details/ncaa-bb-public/ncaa-basketball