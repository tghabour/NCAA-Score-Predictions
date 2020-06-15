
import requests
import pandas as pd
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

##############################################################################

def _get_columns(games_frame):
    """
    Helper function that distinguishes between columns to import and columns to
    ignore when pulling in primary, historical games/statistics data.
    """

    columns_to_ignore = ([

    'created',
    'coverage',
    'status',
    'attendance',
    'possession_arrow',
    'gametime',
    'venue_id',
    'venue_city',
    'venue_state',
    'venue_address',
    'venue_zip',
    'venue_country',
    'venue_name',
    'venue_capacity',
    'tournament_type',
    'tournament_round',
    'tournament_game_no',
    'h_market',
    'h_minutes',
    'h_id',
    'h_league_id',
    'h_rank',
    'h_league_name',
    'h_league_alias',
    'h_conf_id',
    'h_conf_name',
    'h_division_id',
    'h_division_name',
    'h_logo_large',
    'h_logo_medium',
    'h_logo_small',
    'h_ejections',
    'h_flagrant_fouls',
    'h_player_tech_fouls',
    'h_team_tech_fouls',
    'h_coach_tech_fouls',
    'a_market',
    'a_minutes',
    'a_id',
    'a_league_id',
    'a_rank',
    'a_league_name',
    'a_league_alias',
    'a_conf_id',
    'a_conf_name',
    'a_division_id',
    'a_division_name',
    'a_logo_large',
    'a_logo_medium',
    'a_logo_small',
    'a_ejections',
    'a_flagrant_fouls',
    'a_player_tech_fouls',
    'a_team_tech_fouls',
    'a_coach_tech_fouls'])

    columns_to_use = [col for col in games_frame.columns \
                      if col not in columns_to_ignore]

    return columns_to_use

##############################################################################

def _get_heights_inches(height):
    """
    Helper function that converts input heights (provided as strings of the
    form "feet-inches") to total heigh in inches.

    ARGS:
        height: string of the form "x-y", where x is feet and y is inches

    RETURNS:
        total: height in inches (int)
    """

    try:
        feet = height.split('-')[0]
        inches = height.split('-')[1]

        total = (int(feet) * 12) + int(inches)

        return total
    except:
        pass

##############################################################################

def _get_avg_data():
    """
    Function to retreive preliminary average game statistics for 2016 and 2017
    seasons.
    """

    data_path = '../Data/'
    avg_filename = 'NCAA_averges_with_index.csv'

    avg_file = data_path + avg_filename

    averages_df = pd.read_csv(avg_file)
    averages_df.season = averages_df.season.astype(str)

    averages_df = averages_df.dropna(subset = ['game_id'])
    averages_df = averages_df.drop(averages_df.columns[0], axis=1)
    averages_df = averages_df.fillna(0)

    averages_df = _recalc_stats(averages_df)

    return averages_df

##############################################################################

def _make_pct(input_df, input_stat):
    """
    Helper function used to calculate accurate percentages for certain
    statistics after averaging/aggregating data.

    For example, replace average field goal percentages with total field goals
    made divided by total field goals attempted.

    ARGS:
        input_df: specifies aggregated pandas dataframe on which to operate
        input_stat: specifies which performance statistic to reculculate (e.g.,
                    "field_goals_pct", "three_points_pct", etc.)
        input_side: specifies whether calculation pertains to "home" or "away"
                    team

    RETURNS:
        stat_pct: pandas series with properly calculated aggregate avg metric
    """

    stat_numtr = input_df[input_stat + '_made']
    stat_denom = input_df[input_stat + '_att']

    stat_pct = stat_numtr / stat_denom * 100

    return stat_pct

##############################################################################

def _recalc_stats(prelim_averages_df):
    """
    Helper funtion that properly calculates averages for certain statistics
    (namely, percentages) after historical game data has been transformed.
    """

    recalc_list = ['field_goals', 'three_points', 'two_points', 'free_throws']

    final_averages_df = prelim_averages_df

    for stat in recalc_list:
        final_averages_df[stat + '_pct'] = (
            _make_pct(prelim_averages_df, stat))

    final_averages_df['assists_turnover_ratio'] = (
        prelim_averages_df['assists'] /
        prelim_averages_df['turnovers'])

    # Eliminate features we know to have very strong multicollinearioty with others.
    # For instance, 'free_throws_made' has very strong linear relationship with
    # 'free_throws_att'
    final_averages_df = (prelim_averages_df.drop(columns = ['points',
                                                            'rebounds',
                                                            'field_goals_made',
                                                            'free_throws_made',
                                                            'three_points_made',
                                                            'two_points_made',
                                                            'offensive_rebounds',
                                                            'assists',
                                                            'times_tied',
                                                            'field_goals_pct']))

    return final_averages_df

##############################################################################

def scrape_single_roster(team_url, season):
    """
    Retreives the average height abd weight for a given team's roster
    for a given season.
    """

    url = team_url + str(season) + '.html'

    response = requests.get(url)

    if response.status_code != 200:
        return (None, None)

    roster = pd.read_html(url)[0]
    roster.Height = roster.Height.apply(lambda x: _get_heights_inches(x))

    avg_weight = roster.Weight.mean()
    avg_height = roster.Height.mean()

    return (avg_weight, avg_height)

##############################################################################

def get_schools():
    """
    Retreives table of all NCAA men's basketball teams (and locations) from
    www.sports-reference.com.
    """

    url = 'https://www.sports-reference.com/cbb/schools/'
    response = requests.get(url)

    assert response.status_code == 200, 'Unable to retrieve list of schools.'

    scraped_schools_df = pd.read_html(url)[0][['School', 'City, State']]

    return scraped_schools_df

##############################################################################

def get_game_data():
    """
    Retreives saved NCAA game data for 2016-2017 and 2017-2018 seasons and
    returns as pandas DataFrame.

    Only retrieves information for Division I, non-tournament games, played on
    non-neutral sites.

    Original "full" dataset obtained from Google Cloud Platform. For more
    information, refer to:

        https://console.cloud.google.com/marketplace/details/ncaa-bb-public/
                ncaa-basketball
    """

    data_path = '../Data/Raw/'
    game_filename = 'NCAA_games.csv'

    game_file = data_path + game_filename

    games_full     = pd.read_csv(game_file)

    iter_csv = pd.read_csv(game_file, usecols = _get_columns(games_full), \
                           iterator=True, chunksize=1000)

    game_data = (pd.concat([chunk[(pd.isna(chunk['tournament'])) &
                           (chunk['neutral_site'] == False) &
                           (chunk['h_division_alias'] == 'D1') &
                           (chunk['a_division_alias'] == 'D1')]
                            for chunk in iter_csv]))

    game_data.season = game_data.season.astype(str)

    game_data = (game_data.drop(columns = ['h_conf_alias',
                                           'a_conf_alias',
                                           'h_division_alias',
                                           'a_division_alias',
                                           'conference_game',
                                           'tournament',
                                           'neutral_site']))

    return game_data

##############################################################################

def calc_averages(game_data_df, season):
    """
    Calculates average performance statistics for games preceding
    prediction game for each team/season.

    ARGS:
        game_data_df: specifies aggregated pandas dataframe on which to operate
        season: specifies which season over which to calculate averages

    RETURNS:
        season_averages: pandas dataframe with average stats for preceding games
    """

    season_averages = game_data_df[game_data_df.season == season]
    season_averages = (season_averages.sort_values(['alias', 'scheduled_date'])
                                   .copy())

    for school in season_averages.alias.unique():

        school_df = (game_data_df[game_data_df.alias == school]
                     .sort_values(['scheduled_date']))

        for index, row in school_df.iterrows():
            school_row_means = (school_df[(school_df.scheduled_date <
                                row['scheduled_date'])].mean())

            for idx in school_row_means.index:
                season_averages.loc[index, idx] = school_row_means[idx]

    return season_averages

##############################################################################

def get_model_data():
    """
    Function to retrieve processed data for use in linear regression modeling.
    """

    averages = _get_avg_data()

    home_averages = averages[averages.at_home == True]
    away_averages = averages[averages.at_home == False]

    # Sync up team averages into matchups
    model_data = home_averages.merge(away_averages, left_on='game_id',
                                                    right_on='game_id',
                                                    suffixes=('_home', '_away'))

    # Ignore fields not applicable to linear regression (names, conference, etc)
    model_data = (model_data.drop(columns = ['at_home_home',
                                             'at_home_away',
                                             'name_home',
                                             'name_away',
                                             'season_home',
                                             'season_away',
                                             'scheduled_date_away']))

    model_data = (model_data.rename(columns =
                    {'points_game_home': 'ppg_home',
                     'points_game_away': 'ppg_away',
                     'scheduled_date_home': 'scheduled_date'}))

    return model_data

##############################################################################

def run_lm(data, test = False, deg = 1, int_only_ = False, reg = None, \
           alpha_ = None, id_ = ''):

    """
    Function to automate running multiple models at once.
    """

    X_tr, X_vl, X_test, y_tr, y_vl, y_test = (data[0], data[1], data[2],
                                              data[3], data[4], data[5])

    if reg == 'Lasso':
        lm = Lasso(alpha = alpha_)
    elif reg == 'Ridge':
        lm = Ridge(alpha = alpha_)
    else:
        lm = LinearRegression()

    poly = PolynomialFeatures(deg, int_only_)

    X_tr   = poly.fit_transform(X_tr.values)
    X_vl   = poly.transform(X_vl.values)
    X_test = poly.transform(X_test.values)

    # Scale inputs, as applicable
    if reg:
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_vl   = scaler.transform(X_vl)
        X_test = scaler.transform(X_test)

    lm.fit(X_tr, y_tr)

    description = f'{id_}: Degree = {deg} (interaction_only = {int_only_}), ' \
                + f'Regularization = {reg} (alpha = {alpha_})'

    print(description)

    if test:
        RMSE_test = np.sqrt(mean_squared_error(y_test,lm.predict(X_test)))
        print(f'  * Test R^2:  {lm.score(X_test, y_test):.3f}')
        print(f'  * Test RMSE: {RMSE_test}\n')
    else:
        RMSE_tr = np.sqrt(mean_squared_error(y_tr,lm.predict(X_tr)))
        RMSE_vl = np.sqrt(mean_squared_error(y_vl,lm.predict(X_vl)))
        print(f'  * Training R^2:    {lm.score(X_tr, y_tr):.5f}')
        print(f'  * Validation R^2:  {lm.score(X_vl, y_vl):.5f}')
        print(f'  * Training RMSE:   {RMSE_tr:.5f}')
        print(f'  * Validation RMSE: {RMSE_vl:.5f}\n')

    return lm

##############################################################################

def plot_results(results, side):
    """
    Produces three scatter plots summarizing model predictions and residuals:
        1) Actual vs. Predicted
        2) Actual vs. Residuals
        3) Predicted vs. Residuals

    "Side" input specifies whether plots should be provided for "home" team,
    "away" team, or "both".
    """

    plt.figure(figsize=(20,5))

    range1 = [40, 115]
    range2 = [-35, 35]

    ax1 = plt.subplot(1, 3, 1)
    ax1.set_xlim(range1)
    ax1.set_ylim(range1)
    plt.xlabel("ACTUAL")
    plt.ylabel("PREDICTED")

    ax2 = plt.subplot(1, 3, 2)
    ax2.set_xlim(range1)
    ax2.set_ylim(range2)
    plt.xlabel("ACTUAL")
    plt.ylabel("RESIDUAL")

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_xlim(range1)
    ax3.set_ylim(range2)
    plt.xlabel("PREDICTED")
    plt.ylabel("RESIDUAL")

    if side == 'both':
        ax1.scatter(results['actual_home'], results['predicted_home'], alpha = .2)
        ax1.scatter(results['actual_away'], results['predicted_away'], alpha = .2)
        ax1.legend(('Home', 'Away'), loc = 4)


        ax2.scatter(results['actual_home'], results['residual_home'], alpha = .2)
        ax2.scatter(results['actual_away'], results['residual_away'], alpha = .2)
        ax2.legend(('Home', 'Away'), loc = 4)


        ax3.scatter(results['predicted_home'], results['residual_home'], alpha = .2)
        ax3.scatter(results['predicted_away'], results['residual_away'], alpha = .2)
        ax3.legend(('Home', 'Away'), loc = 4)

    else:
        ax1.scatter(results['actual_'+side], results['predicted_'+side], alpha = .2)
        ax2.scatter(results['actual_'+side], results['residual_'+side], alpha = .2)
        ax3.scatter(results['predicted_'+side], results['residual_'+side], alpha = .2)

    return
