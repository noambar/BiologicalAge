import copy

from LabData.DataLoaders.Loader import LoaderData, Loader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.ABILoader import ABILoader
from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataLoaders.ItamarSleepLoader import ItamarSleepLoader
from LabData.DataLoaders.UltrasoundLoader import UltrasoundLoader
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from LabData.DataLoaders.CGMLoader import CGMLoader
from LabData.DataLoaders.DietLoggingLoader import DietLoggingLoader
from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader
from LabData.DataLoaders.LifeStyleLoader import LifeStyleLoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader

from LabData.DataMergers.DataMerger import DataMerger

from LabUtils.Utils import mkdirifnotexists, print_current_time, add_text_at_corner
import os
from LabData.DataAnalyses.KnowledgeGraph.loader_defs import _pickle_obj
import pandas as pd
import numpy as np
import re
from LabData.DataAnalyses.TenK_Trajectories.utils import screen_samples_relative_to_research_stage, \
    add_sex_gender_to_df, screen_by_date, calculate_exact_age, get_medications_prior_to_stage, time_to_seconds
from LabData.DataAnalyses.TenK_Trajectories.utils import get_baseline_medications, get_diet_logging_around_stage, \
    get_baseline_medical_conditions, moving_average
from LabData.DataAnalyses.TenK_Trajectories.archive.defs import BM_COLS_LIST, ARTERY_ULTRASOUND_COLS_LIST
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, levene
from mne.stats import fdr_correction
from scipy.spatial.distance import pdist
import mantel
from itertools import combinations, combinations_with_replacement
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.manifold import TSNE
import umap
from statsmodels.discrete.discrete_model import Logit

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from sklearn.impute import KNNImputer




work_dir = mkdirifnotexists(os.path.join('/net/mraid08/export/genie/LabData/Analyses/10K_Trajectories/biological_age'))
Xs_dir = mkdirifnotexists(os.path.join(work_dir, 'Xs'))
Ys_dir = mkdirifnotexists(os.path.join(work_dir, 'Ys'))
pred_dir = mkdirifnotexists(os.path.join(work_dir, 'predictions'))
mantel_dir = mkdirifnotexists(os.path.join(work_dir, 'mantel_test'))
sex_spec_corrs_dir = mkdirifnotexists(os.path.join(work_dir, 'sex_specific_corrs'))
pred_residuals_fig_dir = mkdirifnotexists(os.path.join(work_dir, 'predictions', 'prediction_residuals'))

norm_dist_capping = {'sample_size_frac': 0.95, 'clip_sigmas': 5, 'remove_sigmas': 8}

# loader_names = ['abi', 'blood_tests', 'body_measures', 'carotid_ultrasound', 'cgm', 'dxa', 'itamar_sleep',
#                 'liver_ultrasound', 'microbiome', 'retina_scan', 'serum_met', 'diet', 'medical_conditions', 'medications']

BM_COLS_LIST = ['waist', 'hips', 'body_fat', 'sitting_blood_pressure_diastolic', 'lying_blood_pressure_diastolic',
                'standing_one_min_blood_pressure_systolic', 'standing_three_min_blood_pressure_diastolic',
                'lying_blood_pressure_systolic', 'standing_one_min_blood_pressure_diastolic',
                'standing_three_min_blood_pressure_systolic', 'standing_three_min_blood_pressure_pulse_rate',
                'bmr', 'hand_grip_left', 'trunk_fat', 'weight', 'bmi', 'height',
                'standing_one_min_blood_pressure_pulse_rate', 'hand_grip_right', 'sitting_blood_pressure_pulse_rate',
                'lying_blood_pressure_pulse_rate', 'sitting_blood_pressure_systolic']

diet_cols = ['add_salt_to_food', 'bread_slices_week', 'bread_type_mainly_eat', 'cereal_type', 'cereals_bowels_week',
             'cheese_fat_percentage_how',
             'cheese_milk_products', 'coffee_cups_day', 'coffee_type', 'cooked_veg_tablespoons_day',
             'diet_major_changes_5years', 'diet_vary_week_to_week',
             'eat_beef', 'eat_cereals_how', 'eat_cheese', 'eat_chicken_poultry', 'eat_kosher', 'eat_lamb_mutton',
             'eat_margarine', 'eat_moldy_cheese_how', 'eat_oily_fish',
             'eat_pork', 'eat_processed_meat', 'eatother_fish', 'fresh_fruit_day',
             'oil_press_type_frying', 'oil_type_frying', 'raw_veg_tablespoons_day', 'spread_type', 'vegeterian_yes_no',
             'water_glasses_day']
                # last_ate_meat_age_age
smoking_cols = ['smoke_houshold', 'smoke_tobacco_now', 'tobacco_past_how_often']
exercise_cols = ['climb_staires_tymes_a_day', 'high_exercise_duration', 'high_exercise_times_a_month',
                 'moderate_activity_minutes',
                 'physical_activity_maderate_days_a_week', 'usual_walking_pace', 'vigorous_activity_minutes',
                 'walking_10min_days_a_week', 'walking_minutes_day']
alcohol_cols = ['alcohol_drink', 'beer_cider_pints_week', 'drink_alcohol_with_meals',
                'fortified_wine__glasses_week', 'liqueurs_measures_week',
                'other_alcoholic_glasses_week', 'white_wine_glasses_week', 'why_stop_drinking']  # 'red_wine_glasses_week'
working_cols = ['manual_physical_work', 'shift_work', 'transportaion_to_work', 'work_days_a_week', 'work_hours',
                'work_hours_day']
sleeping_cols = ['easy_getting_up', 'falling_asleep_during_daytime', 'nap_during_day', 'sleep_hours_in_24H',
                 'snoring', 'trouble_falling_a_sleep']
habits_cols = ['consider_yourself_morning_evening', 'drive_faster_often', 'hours_driving', 'hours_outdoors_summer',
               'hours_outdoors_winter', 'hours_using_computer_not_work', 'hours_watching_tv',
               'mobile_phone_side_of_head', 'mobile_phone_use_duration_per_week', 'pet_past', 'pet_present',
               'transport_forms_used', 'visit_friends_family_times', 'years_using_mobile_phone']
household_cols = ['people_living_together', 'people_living_together_retalated']
answer_mapping_dic = {'2-3 פעמים בשבוע': 'Two or three times a week',
                      '4-5 פעמים בשבוע': 'Four to five times a week',
                      '2-3 פעמים בחודש האחרון': 'Two or three times in the last month',
                      'אף פעם/לעיתים רחוקות': 'Never',
                      'בדרך כלל': 'Usually',
                      'אף אחד מהנ"ל': 'None of the above',
                      'אף פעם': 'Never',
                      'Not / rarely': 'Not or rarely',
                      'Never / rare': 'Never or rare',
                      'Car / motorcycle': 'Car or motorcycle',
                      'פחות מפעם בשבוע': 'Less than once a week',
                      'פעם אחת או יותר ביום': 'Once a week or more',
                      '5-6 פעמים בשבוע': 'Five to six times a week'
                     }

smoke_q = ['smoke_houshold', 'smoke_tobacco_now', 'tobacco_past_how_often']
sleep_q = ['consider_yourself_morning_evening', 'easy_getting_up', 'falling_asleep_during_daytime', 'nap_during_day',
           'sleep_hours_in_24H', 'snoring', 'trouble_falling_a_sleep']
physical_activity_q = ['drive_faster_often', 'hours_driving', 'hours_using_computer_not_work', 'hours_watching_tv',
                       'transport_forms_used', 'climb_staires_tymes_a_day', 'high_exercise_duration',
                       'high_exercise_times_a_month', 'moderate_activity_minutes',
                       'physical_activity_maderate_days_a_week', 'usual_walking_pace', 'vigorous_activity_minutes',
                       'walking_10min_days_a_week', 'walking_minutes_day']
sun_exposure_q  = ['hours_outdoors_summer', 'hours_outdoors_winter']
electronic_device_use_q = ['mobile_phone_side_of_head', 'mobile_phone_use_duration_per_week', 'years_using_mobile_phone']
social_support_q = ['visit_friends_family_times']
other_q = ['pet_past', 'pet_present']
employment_q = ['manual_physical_work', 'shift_work', 'transportaion_to_work', 'work_days_a_week', 'work_hours',
                'work_hours_day']
alcohol_q = ['alcohol_drink', 'beer_cider_pints_week', 'drink_alcohol_with_meals', 'fortified_wine__glasses_week',
             'liqueurs_measures_week', 'other_alcoholic_glasses_week', 'white_wine_glasses_week', 'why_stop_drinking']


# modality_name_mapping = {'serum_met': 'lipidomics', 'itamar_sleep': 'sleep', 'blood_tests': 'blood tests',
#                         'medical_conditions': 'medical conditions', 'carotid_ultrasound': 'carotid ultrasound',
#                         'abi': 'ABI', 'retina_scan': 'eye images', 'cgm': 'CGM', 'dxa': 'DXA',
#                          'liver_ultrasound': 'liver ultrasound', 'body_measures': 'anthropometrics'}

modality_name_mapping = {'blood_lipids': 'Blood lipids', 'body_composition': 'Body composition',
                         'bone_density': 'Bone density', 'diet': 'Diet', 'diet_questions': 'Diet questionnaires',
                         'frailty': 'Frailty',
                         'glycemic_status': 'Insulin resistance', 'liver': 'Liver health',
                         'microbiome': 'Gut microbiome', 'sleep': 'Sleep characteristics',
                         'cardiovascular_system': 'Cardiovascular system', 'immune_system': 'Immune system',
                         'lifestyle': 'Lifestyle', 'medical_conditions': 'Medical diagnosis',
                         'medications': 'Medications', 'renal_function': 'Renal function',
                         'hematopoietic_system': 'Hematopoietic system'}

# dominant_hand, is_getting_period, dizziness, on_hormone_therapy

rscv_kwargs = {'model': 'lightgbm',
               'n_cols_per_job': 1,
               'n_random': 10,
               'random_seed': 0,
               'standadize_Y': 'False',
               'cross_validation': 'True',
               'test_size': 0.3,
               'bootstrap': 'False',
               'n_bootstraps': 100}

male_color = 'dodgerblue'
female_color = 'tomato'


def q_setup():
    from LabUtils.addloglevels import sethandlers
    from LabData import config_global as config

    os.chdir(work_dir)

    from LabUtils.addloglevels import handlers_were_set
    if not handlers_were_set:
        sethandlers(file_dir=work_dir)
    return config

def clean_and_save(loader, name=None):
    print(name)
    loader = screen_by_date(loader)
    loader = add_sex_gender_to_df(loader)
    loader.df = transform_object_to_dummies(loader.df)
    loader.df.rename(columns={k: k.replace("'", '').replace(",", ' ') for k in loader.df.columns}, inplace=True)
    age_y = loader.df[['age']]
    age_y = age_y[(age_y['age'] <= 75) & (age_y['age'] >= 40)].dropna()
    x = loader.df.drop('age', axis=1).reset_index('Date', drop=True)
    y = age_y.reset_index('Date', drop=True)
    x = x.reindex(y.index).dropna(how='all')
    y = y.reindex(x.index).dropna(how='all')
    # _pickle_obj(x, os.path.join(Xs_dir, '%s.pkl' % name))
    # _pickle_obj(y, os.path.join(Ys_dir, '%s.pkl' % name))
    x.to_csv(os.path.join(Xs_dir, '%s.csv' % name))
    x[x['gender'] == 0].drop('gender', axis=1).to_csv(os.path.join(Xs_dir, '%s_female.csv' % name))
    x[x['gender'] == 1].drop('gender', axis=1).to_csv(os.path.join(Xs_dir, '%s_male.csv' % name))
    x.to_csv(os.path.join(Xs_dir, '%s.csv' % name))
    y.to_csv(os.path.join(Ys_dir, '%s.csv' % name))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    males = y.reindex(x[x['gender'] == 1].index)['age'].dropna()
    females = y.reindex(x[x['gender'] == 0].index)['age'].dropna()
    ax.hist([males, females], bins=15, label=['male n=%d' % males.shape[0], 'female n=%d' % females.shape[0]])
    t, p = mannwhitneyu(males, females)
    ax.set_xlabel('Age', fontsize=15)
    ax.set_ylabel('Number of participants', fontsize=15)
    ax.set_title('%s, p=%0.2g' % (name, p), fontsize=20)
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig(os.path.join(Ys_dir, '%s_distribution_by_gender.png' % name), dpi=200)

    # match males and females on age
    males, females = match_on_age(males, females)
    x.loc[females.index].drop('gender', axis=1).to_csv(os.path.join(Xs_dir, '%s_female_equal.csv' % name))
    x.loc[males.index].drop('gender', axis=1).to_csv(os.path.join(Xs_dir, '%s_male_equal.csv' % name))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist([males, females], bins=15, label=['male n=%d' % males.shape[0], 'female n=%d' % females.shape[0]])
    t, p = mannwhitneyu(males, females)
    ax.set_xlabel('Age', fontsize=15)
    ax.set_ylabel('Number of participants', fontsize=15)
    ax.set_title('%s, p=%0.2g' % (name, p), fontsize=20)
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig(os.path.join(Ys_dir, '%s_distribution_by_gender_equal.png' % name), dpi=200)


def match_on_age(males, females, max_tol=5, min_tol=1e-5, num=7):
    def isclose(males, females, tol=1):
        result = np.where( np.triu(np.isclose(males.values[:, None],
                                              females.values[None, :], rtol=0.0, atol=tol, ), 1))
        pairs = np.swapaxes(result, 0, 1)
        pairs = pd.DataFrame(pairs, columns=['male', 'female'])
        pairs = pairs.groupby('male').first().reset_index()
        pairs = pairs.groupby('female').first().reset_index()
        return males.iloc[pairs['male']], females.iloc[pairs['female']]

    tols = np.linspace(start=min_tol, stop=max_tol, num=num)
    males_final, females_final = [], []
    for tol in tols:
        males1, females1 = isclose(males, females, tol)
        # print(males1.shape)
        males = males.drop(males1.index, axis=0)
        females = females.drop(females1.index, axis=0)
        males_final.append(males1)
        females_final.append(females1)
    males1 = pd.concat(males_final)
    females1 = pd.concat(females_final)
    return males1, females1



def transform_object_to_dummies(df):
    object_cols = []
    numeric_cols = []
    for c in df.columns:
        try:
            df[c].astype(float)
            numeric_cols.append(c)
        except:
            object_cols.append(c)
    if len(object_cols) > 0:
        df = pd.concat((df.loc[:, numeric_cols], pd.get_dummies(df.loc[:, object_cols])), axis=1)
    return df.astype(float)

    # object_cols = df.dtypes == object
    # non_object_cols = df.dtypes != object
    # if object_cols.sum() > 0:
    #     df = pd.concat((df.loc[:, non_object_cols], pd.get_dummies(df.loc[:, object_cols])), axis=1)
    # return df.astype(float)


def adjust_index(loader):
    loader.df[['RegistrationCode', 'Date']] = loader.df_metadata[['RegistrationCode', 'Date']]
    loader.df.set_index(['RegistrationCode', 'Date'], inplace=True)
    loader.df_metadata.set_index(['RegistrationCode', 'Date'], inplace=True)
    return loader


def fix_date_dtype(loader):
    index_names = loader.df.index.names
    loader.df = loader.df.groupby(index_names).first()
    loader.df_metadata = loader.df_metadata.groupby(index_names).first()
    assert 'Date' in index_names
    loader.df.reset_index(inplace=True)
    loader.df_metadata.reset_index(inplace=True)
    loader.df['Date'] = pd.to_datetime(loader.df['Date'], errors='coerce').astype('datetime64[ns]')
    loader.df.dropna(subset=['Date'], inplace=True)
    loader.df_metadata['Date'] = pd.to_datetime(loader.df_metadata['Date'], errors='coerce').astype('datetime64[ns]')
    loader.df_metadata.dropna(subset=['Date'], inplace=True)
    # loader.df['Date'] = loader.df['Date'].astype('datetime64[ns]')
    # loader.df_metadata['Date'] = loader.df_metadata['Date'].astype('datetime64[ns]')
    loader.df.set_index(index_names, inplace=True)
    loader.df_metadata.set_index(index_names, inplace=True)
    return loader


def add_date_from_body_measures(loader, body_measures):
    loader.df = loader.df.join(body_measures.df.reset_index('Date')[['Date']], how='left').reset_index()
    loader.df.set_index(['RegistrationCode', 'Date'], inplace=True)
    loader.df_metadata.index = loader.df.index
    return loader


def add_yob_from_subject_loader(loader):
    from LabData.DataLoaders.SubjectLoader import SubjectLoader
    sl = SubjectLoader().get_data(reg_ids=loader.df_metadata.RegistrationCode)
    index_name = loader.df_metadata.index.names

    loader.df_metadata = loader.df_metadata.reset_index().set_index('RegistrationCode')\
        .drop(['yob', 'month_of_birth'], axis=1).join(sl.df.groupby('RegistrationCode')
                                                      .first()['yob', 'month_of_birth']).reset_index()\
        .set_index(index_name)
    return loader


def merge_data_loaders(loaders):
    def _merge_pair(l1, l2):
        dm = DataMerger([l1, l2])
        # do a merge on the left loader, and keep only a single record per reg_id with the closest date
        tenk_data = dm.get_x(how='left', res_index_names=['RegistrationCode', 'Date'],
                             inexact_index='Date', inexact_index_direction='nearest',
                             inexact_index_tolerance=pd.Timedelta(days=365))
        return tenk_data

    tenk_data = loaders[0]
    for other_loader in loaders[1:]:
        cols = other_loader.df.columns
        for col in cols:
            temp_loader = copy.deepcopy(other_loader)
            temp_loader.df = temp_loader.df.drop([c for c in cols if c != col], axis=1).dropna()
            temp_loader.df_metadata = temp_loader.df_metadata.loc[temp_loader.df.index]
            tenk_data = _merge_pair(tenk_data, temp_loader)

    return tenk_data


def build_Xs_and_Ys():
    body_measures = BodyMeasuresLoader().get_data(study_ids=['10K'], research_stage=['baseline'], groupby_reg='first',
                                                  min_col_present_frac=0.5)
    #
    # # microbiome
    # tenk_data = GutMBLoader().get_data('segal_species', study_ids=['10K'], groupby_reg='first',
    #                                    research_stage=['baseline'], min_col_present=500, min_col_val=1e-4,
    #                                    min_reads=3, take_log=True)
    # # tenk_data = add_yob_from_subject_loader(tenk_data)
    # tenk_data = adjust_index(tenk_data)
    # clean_and_save(tenk_data, 'microbiome')
    #
    # # Diet - short food names
    # dl = DietLoggingLoader()
    # tenk_data = dl.get_data(study_ids=['10K'])
    # tenk_data.df = get_diet_logging_around_stage(tenk_data.df, stage='baseline', delta_before=2, delta_after=14)
    # tenk_data = dl.daily_mean_food_consumption(df=tenk_data.df, kcal_limit=500, min_col_present_frac=0.05,
    #                                            level='shortname_eng')
    # tenk_data.df = tenk_data.df.astype(float)
    # tenk_data.df.replace({0: np.nan}, inplace=True)
    # tenk_data.df = Loader._norm_dist_capping(tenk_data.df, {'sample_size_frac': 0.99, 'clip_sigmas': 10})
    # tenk_data.df.fillna(0, inplace=True)
    # tenk_data.df_metadata = tenk_data.df_metadata.reindex(tenk_data.df.index)
    # tenk_data = add_date_from_body_measures(tenk_data, body_measures)
    # clean_and_save(tenk_data, 'diet')
    #
    # # # diet questionnaire
    # tenk_data = LifeStyleLoader().get_data(study_ids=['10K'], min_col_present=5000, df='english', groupby_reg='first',
    #                                        cols=diet_cols)
    # tenk_data.df.dropna(how='all', inplace=True)
    # tenk_data.df.replace(answer_mapping_dic, inplace=True)
    # tenk_data.df = tenk_data.df.applymap(lambda x: x.replace('/', 'or') if isinstance(x, str) else x)
    # tenk_data.df_metadata = tenk_data.df_metadata.loc[tenk_data.df.index]
    # tenk_data = fix_date_dtype(tenk_data)
    # clean_and_save(tenk_data, 'diet_questions')
    #
    # #
    # # sleep
    # tenk_data = ItamarSleepLoader().get_data(study_ids=['10K'],  min_col_present_frac=0.5, groupby_reg='median',
    #                                          research_stage=['baseline'])
    # tenk_data = add_date_from_body_measures(tenk_data, body_measures)
    # tenk_data.df = tenk_data.df.loc[:, ~tenk_data.df.columns.duplicated()]
    # if 'StudyStartTime' in tenk_data.df.columns:
    #     tenk_data.df.drop(['StudyStartTime'], axis=1, inplace=True)
    # if 'StudyEndTime' in tenk_data.df.columns:
    #     tenk_data.df.drop(['StudyEndTime'], axis=1, inplace=True)
    # clean_and_save(tenk_data, 'sleep')

    # liver
    tenk_data = UltrasoundLoader().get_data(study_ids=['10K'], groupby_reg='first', compute_ys=True)
    cols = ['bt__alt_gpt', 'bt__ast_got', 'bt__alkaline_phosphatase', 'bt__albumin', 'bt__protein_total',
            'bt__bilirubin_total', 'bt__platelets']
    tenk_data_bt = BloodTestsLoader().get_data(reg_ids=tenk_data.df.index.get_level_values('RegistrationCode'),
                                               cols=cols,
                                               min_col_present_frac=0.2, norm_dist_capping=norm_dist_capping)
    tenk_data_bt.df.dropna(how='all', inplace=True)
    tenk_data_bt.df_metadata = tenk_data_bt.df_metadata.loc[tenk_data_bt.df.index]
    tenk_data_bt = fix_date_dtype(tenk_data_bt)
    tenk_data = merge_data_loaders([tenk_data, tenk_data_bt])
    clean_and_save(tenk_data, 'liver')

    # insulin resistance
    # CGM
    tenk_data = CGMLoader().get_data(study_ids=['10K'])
    iglu_df = pd.read_csv('/net/mraid08/export/genie/LabData/Analyses/ayyak/CGM/iglu/iglu_days_2_13.csv', index_col=0)
    iglu_df['RegistrationCode'] = [s.split('/')[0] for s in iglu_df['id']]
    iglu_df['ConnectionID'] = [s.split('/')[-1] for s in iglu_df['id']]
    iglu_df = iglu_df.groupby('RegistrationCode').first().reset_index()
    iglu_df = iglu_df.set_index(['RegistrationCode', 'ConnectionID']).drop('id', axis=1)
    tenk_data.df_metadata = tenk_data.df_metadata.reindex(iglu_df.index).dropna(how='all')
    tenk_data.df_metadata['Date'] = pd.to_datetime([d.date() for d in pd.to_datetime(tenk_data.df_metadata['Period_start'])])
    tenk_data.df_metadata = tenk_data.df_metadata.reset_index().set_index(['RegistrationCode', 'Date'])
    tenk_data.df = iglu_df
    tenk_data.df.index = tenk_data.df_metadata.index

    cols = ['bt__glucose', 'bt__hba1c']
    tenk_data_bt = BloodTestsLoader().get_data(reg_ids=tenk_data.df.index.get_level_values('RegistrationCode'),
                                               cols=cols, norm_dist_capping=norm_dist_capping)
    tenk_data_bt.df.dropna(how='all', inplace=True)
    tenk_data_bt.df_metadata = tenk_data_bt.df_metadata.loc[tenk_data_bt.df.index]
    tenk_data_bt = fix_date_dtype(tenk_data_bt)
    tenk_data = merge_data_loaders([tenk_data, tenk_data_bt])
    clean_and_save(tenk_data, 'glycemic_status')

    # DXA
    tenk_data = DEXALoader().get_data(study_ids=['10K'], groupby_reg='first', min_col_present_frac=0.7,
                                      norm_dist_capping=norm_dist_capping)
    tenk_data.df.drop(tenk_data.df.filter(regex='z_score|t_score|ya_percent|am_percent|dose').columns, 1, inplace=True)
    tenk_data.df_columns_metadata = tenk_data.df_columns_metadata[tenk_data.df_columns_metadata['column_name']
        .isin(tenk_data.df.columns)]

    # bone density
    col_type = ['Area', 'Bone Mineral Content', 'Bone Mineral Density', 'Width', 'height']
    temp_loader = copy.deepcopy(tenk_data)
    temp_loader.df = temp_loader.df.reindex(tenk_data.df_columns_metadata[tenk_data.df_columns_metadata['Type']
                                            .isin(col_type)]['column_name'], axis=1).dropna(how='all', axis=1)
    clean_and_save(temp_loader, 'bone_density')

    # body composition
    col_type = ['VAT Volume', 'VAT Area', 'SAT Volume', 'SAT Mass', 'SAT Area', 'VAT Mass', 'Body Composition']
    tenk_data.df = tenk_data.df.reindex(tenk_data.df_columns_metadata[tenk_data.df_columns_metadata['Type']
                                            .isin(col_type)]['column_name'], axis=1).dropna(how='all', axis=1)

    cols = ['bmi', 'waist', 'weight', 'height', 'hips']
    tenk_data_bt = BodyMeasuresLoader().get_data(reg_ids=tenk_data.df.index.get_level_values('RegistrationCode'),
                                                 cols=cols)
    tenk_data_bt.df.dropna(how='all', inplace=True)
    tenk_data_bt.df_metadata = tenk_data_bt.df_metadata.loc[tenk_data_bt.df.index]
    tenk_data = fix_date_dtype(tenk_data)
    tenk_data_bt = fix_date_dtype(tenk_data_bt)
    tenk_data = merge_data_loaders([tenk_data, tenk_data_bt])
    clean_and_save(tenk_data, 'body_composition')

    # cardiovascular system
    # ABI
    tenk_data = ABILoader().get_data(study_ids=['10K'],  min_col_present_frac=0.2, groupby_reg='first')
    tenk_data.df['from_l_thigh_to_l_ankle_duration'] = tenk_data.df['from_l_thigh_to_l_ankle_duration'].apply(lambda t:
                                                                                                              time_to_seconds(t))
    tenk_data.df['from_r_thigh_to_r_ankle_duration'] = tenk_data.df['from_r_thigh_to_r_ankle_duration'].apply(lambda t:
                                                                                                              time_to_seconds(t))
    # blood pressure
    tenk_data_bp = BodyMeasuresLoader().get_data(reg_ids=tenk_data.df.index.get_level_values('RegistrationCode'))
    tenk_data_bp.df = tenk_data_bp.df.filter(regex='blood_pressure').dropna(how='all')
    tenk_data_bp.df_metadata = tenk_data_bp.df_metadata.loc[tenk_data_bp.df.index]

    # carotid ultrasound
    tenk_data_cu = UltrasoundLoader().get_data(study_ids=['10K'], cols=ARTERY_ULTRASOUND_COLS_LIST)

    # retina images
    tenk_data_rs = RetinaScanLoader().get_data(study_ids=['10K'], max_on_most_freq_val_in_col=0.7)
    tenk_data_rs.df = tenk_data_rs.df.groupby(['RegistrationCode', 'Date']).mean()
    tenk_data_rs.df_metadata = tenk_data_rs.df_metadata.groupby(['RegistrationCode', 'Date']).first()

    # ecg
    tenk_data_ecg = ECGTextLoader().get_data(study_ids=['10K'], min_col_present=5000)
    tenk_data_ecg.df.drop(['qrs', 'conclusion', 'non_confirmed_diagnosis', 'st_t'], axis=1, inplace=True)

    tenk_data = fix_date_dtype(tenk_data)
    tenk_data_bp = fix_date_dtype(tenk_data_bp)
    tenk_data_cu = fix_date_dtype(tenk_data_cu)
    tenk_data_rs = fix_date_dtype(tenk_data_rs)
    tenk_data_ecg = fix_date_dtype(tenk_data_ecg)
    tenk_data = merge_data_loaders([tenk_data, tenk_data_bp, tenk_data_cu, tenk_data_rs, tenk_data_ecg])

    clean_and_save(tenk_data, 'cardiovascular_system')


    # frailty
    # DXA
    cols = ['body_comp_leg_right_lean_mass', 'body_comp_leg_left_lean_mass', 'body_comp_arm_right_lean_mass',
            'body_comp_arm_left_lean_mass']
    tenk_data = DEXALoader().get_data(study_ids=['10K'], groupby_reg='first', min_col_present_frac=0.7,
                                      norm_dist_capping=norm_dist_capping, cols=cols)

    cols = ['hand_grip_left', 'hand_grip_right', 'height']
    tenk_data_bt = BodyMeasuresLoader().get_data(reg_ids=tenk_data.df.index.get_level_values('RegistrationCode'),
                                                 cols=cols)
    tenk_data_bt.df['height'] = tenk_data_bt.df['height'] ** 2
    tenk_data_bt.df.dropna(how='all', inplace=True)
    tenk_data_bt.df_metadata = tenk_data_bt.df_metadata.loc[tenk_data_bt.df.index]
    tenk_data = fix_date_dtype(tenk_data)
    tenk_data_bt = fix_date_dtype(tenk_data_bt)
    tenk_data = merge_data_loaders([tenk_data, tenk_data_bt])
    clean_and_save(tenk_data, 'frailty')


    #
    # blood lipids
    # serum lipidomics
    tenk_data = SerumMetabolomicsLoader().get_data(precomputed_loader_fname='metab_10k_data_RT_clustering',
                                                   min_col_present_frac=0.6) # min_col_present_frac=0.7)
    tenk_data = adjust_index(tenk_data)
    cols = ['bt__hdl_cholesterol', 'bt__non_hdl_cholesterol', 'bt__triglycerides', 'bt__total_cholesterol']
    tenk_data_bt = BloodTestsLoader().get_data(reg_ids=tenk_data.df.index.get_level_values('RegistrationCode'),
                                               cols=cols, norm_dist_capping=norm_dist_capping)
    tenk_data_bt.df.dropna(how='all', inplace=True)
    tenk_data_bt.df_metadata = tenk_data_bt.df_metadata.loc[tenk_data_bt.df.index]
    tenk_data_bt = fix_date_dtype(tenk_data_bt)
    tenk_data = merge_data_loaders([tenk_data, tenk_data_bt])

    clean_and_save(tenk_data, 'blood_lipids')

    # lifestyle
    tenk_data = LifeStyleLoader().get_data(study_ids=['10K'], min_col_present=5000, df='english', groupby_reg='first',
                                           cols=smoking+sleep_q+physical_activity_q+sun_exposure_q+
                                                electronic_device_use_q+social_support_q+other_q+employment_q+alcohol_q)
    tenk_data.df.dropna(how='all', inplace=True)
    tenk_data.df.replace(answer_mapping_dic, inplace=True)
    tenk_data.df = tenk_data.df.applymap(lambda x: x.replace('/', 'or') if isinstance(x, str) else x)
    # tenk_data.df = tenk_data.df.rename(columns={k: k.replace('/', '-') for k in tenk_data.df.columns})
    tenk_data.df_metadata = tenk_data.df_metadata.loc[tenk_data.df.index]
    tenk_data = fix_date_dtype(tenk_data)
    clean_and_save(tenk_data, 'lifestyle')


    # medications
    tenk_data = get_baseline_medications()
    tenk_data = Medications10KLoader().get_data(df=tenk_data.df, pivot_by=4)
    # in case there are multiple reportings, keep a positive one if exists
    tenk_data.df = tenk_data.df.groupby('RegistrationCode').max()
    medication_names = tenk_data.df.columns
    # medication matrix should be complete, adding participants from body measures loader
    # merging loaders to get metadata
    tenk_data = DataMerger([body_measures, tenk_data]).get_x(res_index_names=['RegistrationCode'])
    # keeping only the medication columns
    tenk_data.df = tenk_data.df.loc[:, medication_names]
    tenk_data.df.fillna(0, inplace=True)
    tenk_data.df = tenk_data.df.loc[:, tenk_data.df.sum() >= 10].astype(float)
    tenk_data.df = tenk_data.df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    tenk_data = add_date_from_body_measures(tenk_data, body_measures)
    clean_and_save(tenk_data, 'medications')

    # medical conditions
    tenk_data = get_baseline_medical_conditions()
    tenk_data.df = tenk_data.df.fillna(-1).pivot_table(index=['RegistrationCode'], columns=['medical_condition'],
                                                       values='Start')
    tenk_data.df_metadata = tenk_data.df_metadata.groupby('RegistrationCode').first()
    medical_condition_names = tenk_data.df.columns
    # merging loaders to get metadata
    tenk_data = DataMerger([body_measures, tenk_data]).get_x(res_index_names=['RegistrationCode'])
    # keeping only the medication columns
    tenk_data.df = tenk_data.df.loc[:, medical_condition_names]

    tenk_data.df.fillna(0, inplace=True)
    tenk_data.df.replace({-1: np.nan}, inplace=True)
    tenk_data.df = tenk_data.df.loc[:, tenk_data.df.sum() >= 10].astype(float)
    tenk_data = add_date_from_body_measures(tenk_data, body_measures)
    clean_and_save(tenk_data, 'medical_conditions')

    # immune system
    cols = ['bt__lymphocytes_%', 'bt__lymphocytes_abs', 'bt__neutrophils_%', 'bt__neutrophils_abs', 'bt__monocytes_%',
            'bt__monocytes_abs', 'bt__eosinophils_%', 'bt__eosinophils_abs', 'bt__basophils_%', 'bt__basophils_abs',
            'bt__wbc']

    tenk_data = BodyMeasuresLoader().get_data(study_ids=['10K'], research_stage=['baseline'], groupby_reg='first',
                                                  min_col_present_frac=0.5, cols=['bmi'])
    tenk_data_bt = BloodTestsLoader().get_data(reg_ids=tenk_data.df.index.get_level_values('RegistrationCode'),
                                               cols=cols, norm_dist_capping=norm_dist_capping)
    tenk_data_bt.df.dropna(how='all', inplace=True)
    tenk_data_bt.df_metadata = tenk_data_bt.df_metadata.loc[tenk_data_bt.df.index]
    tenk_data_bt = fix_date_dtype(tenk_data_bt)
    tenk_data = merge_data_loaders([tenk_data, tenk_data_bt])
    tenk_data.df = tenk_data.df.drop('bmi', axis=1).dropna(how='all', axis=0).dropna(how='all', axis=1)
    tenk_data.df_metadata = tenk_data.df_metadata.loc[tenk_data.df.index]
    clean_and_save(tenk_data, 'immune_system')

    # hematopoietic system
    cols = ['bt__mcv', 'bt__hemoglobin', 'bt__hct', 'bt__mchc', 'bt__mch', 'bt__rdw', 'bt__rbc', 'bt__ferritin']

    tenk_data = BodyMeasuresLoader().get_data(study_ids=['10K'], research_stage=['baseline'], groupby_reg='first',
                                                  min_col_present_frac=0.5, cols=['bmi'])
    tenk_data_bt = BloodTestsLoader().get_data(reg_ids=tenk_data.df.index.get_level_values('RegistrationCode'),
                                               cols=cols, norm_dist_capping=norm_dist_capping)
    tenk_data_bt.df.dropna(how='all', inplace=True)
    tenk_data_bt.df_metadata = tenk_data_bt.df_metadata.loc[tenk_data_bt.df.index]
    tenk_data_bt = fix_date_dtype(tenk_data_bt)
    tenk_data = merge_data_loaders([tenk_data, tenk_data_bt])
    tenk_data.df = tenk_data.df.drop('bmi', axis=1).dropna(how='all', axis=0).dropna(how='all', axis=1)
    tenk_data.df_metadata = tenk_data.df_metadata.loc[tenk_data.df.index]
    clean_and_save(tenk_data, 'hematopoietic_system')

    # renal function
    cols = ['bt__creatinine', 'bt__potassium', 'bt__sodium', 'bt__urea']

    tenk_data = BodyMeasuresLoader().get_data(study_ids=['10K'], research_stage=['baseline'], groupby_reg='first',
                                                  min_col_present_frac=0.5, cols=['bmi'])
    tenk_data_bt = BloodTestsLoader().get_data(reg_ids=tenk_data.df.index.get_level_values('RegistrationCode'),
                                               cols=cols, norm_dist_capping=norm_dist_capping)
    tenk_data_bt.df.dropna(how='all', inplace=True)
    tenk_data_bt.df_metadata = tenk_data_bt.df_metadata.loc[tenk_data_bt.df.index]
    tenk_data_bt = fix_date_dtype(tenk_data_bt)
    tenk_data = merge_data_loaders([tenk_data, tenk_data_bt])
    tenk_data.df = tenk_data.df.drop('bmi', axis=1).dropna(how='all', axis=0).dropna(how='all', axis=1)
    tenk_data.df_metadata = tenk_data.df_metadata.loc[tenk_data.df.index]
    clean_and_save(tenk_data, 'renal_function')




def build_Xs_and_Ys_old():
    # # # body measures
    body_measures = BodyMeasuresLoader().get_data(study_ids=['10K'], research_stage=['baseline'], groupby_reg='first',
                                                  min_col_present_frac=0.5)
    body_measures.df.drop(body_measures.df.filter(regex='is_getting_period|on_hormone_therapy').columns, 1, inplace=True)
    clean_and_save(body_measures, 'body_measures')

    # # serum metabolomics
    # tenk_data = SerumMetabolomicsLoader().get_data(precomputed_loader_fname='metab_10k_data_RT_clustering',
    #                                                min_col_present_frac=0.7)
    # tenk_data = adjust_index(tenk_data)
    # clean_and_save(tenk_data, 'serum_met')
    #
    # # ABI
    # tenk_data = ABILoader().get_data(study_ids=['10K'],  min_col_present_frac=0.2, groupby_reg='first')
    # tenk_data.df['from_l_thigh_to_l_ankle_duration'] = tenk_data.df['from_l_thigh_to_l_ankle_duration'].apply(lambda t:
    #                                                                                                           time_to_seconds(t))
    # tenk_data.df['from_r_thigh_to_r_ankle_duration'] = tenk_data.df['from_r_thigh_to_r_ankle_duration'].apply(lambda t:
    #                                                                                                           time_to_seconds(t))
    # clean_and_save(tenk_data, 'abi')
    #
    # # blood tests
    # tenk_data = BloodTestsLoader().get_data(study_ids=['10K'], groupby_reg='first', min_col_present_frac=0.5,
    #                                         research_stage=['baseline'], norm_dist_capping=norm_dist_capping)
    # clean_and_save(tenk_data, 'blood_tests')

    # microbiome
    tenk_data = GutMBLoader().get_data('segal_species', study_ids=['10K'], groupby_reg='first',
                                       research_stage=['baseline'], min_col_present=500, min_col_val=1e-4,
                                       min_reads=3, take_log=True)
    tenk_data = add_yob_from_subject_loader(tenk_data)
    tenk_data = adjust_index(tenk_data)
    clean_and_save(tenk_data, 'microbiome')

    # microbiome full
    tenk_data = GutMBLoader().get_data('segal_species', study_ids=['10K'], groupby_reg='first',
                                       research_stage=['baseline'], min_col_present=0, min_col_val=1e-4,
                                       min_reads=3, take_log=True, col_names_as_ids=True)
    tenk_data = add_yob_from_subject_loader(tenk_data)
    tenk_data = adjust_index(tenk_data)
    clean_and_save(tenk_data, 'microbiome_full')

    # DXA
    tenk_data = DEXALoader().get_data(study_ids=['10K'], groupby_reg='first', min_col_present_frac=0.7,
                                      norm_dist_capping=norm_dist_capping)
    tenk_data.df.drop(tenk_data.df.filter(regex='z_score|t_score|ya_percent|am_percent|dose').columns, 1, inplace=True)
    clean_and_save(tenk_data, 'dxa')

    # sleep
    tenk_data = ItamarSleepLoader().get_data(study_ids=['10K'],  min_col_present_frac=0.5, groupby_reg='median',
                                             research_stage=['baseline'])
    tenk_data = add_date_from_body_measures(tenk_data, body_measures)
    tenk_data.df = tenk_data.df.loc[:, ~tenk_data.df.columns.duplicated()]
    if 'StudyStartTime' in tenk_data.df.columns:
        tenk_data.df.drop(['StudyStartTime'], axis=1, inplace=True)
    if 'StudyEndTime' in tenk_data.df.columns:
        tenk_data.df.drop(['StudyEndTime'], axis=1, inplace=True)
    clean_and_save(tenk_data, 'itamar_sleep')

    # # liver ultrasound
    tenk_data = UltrasoundLoader().get_data(study_ids=['10K'], groupby_reg='first', compute_ys=True)
    clean_and_save(tenk_data, 'liver_ultrasound')
    #
    # carotid ultrasound
    tenk_data = UltrasoundLoader().get_data(study_ids=['10K'], groupby_reg='first', cols=ARTERY_ULTRASOUND_COLS_LIST)
    clean_and_save(tenk_data, 'carotid_ultrasound')

    # retina images
    tenk_data = RetinaScanLoader().get_data(study_ids=['10K'], max_on_most_freq_val_in_col=0.7, groupby_reg='first')
    tenk_data.df = tenk_data.df.groupby(['RegistrationCode', 'Date']).mean()
    tenk_data.df_metadata = tenk_data.df_metadata.groupby(['RegistrationCode', 'Date']).first()
    clean_and_save(tenk_data, 'retina_scan')

    # CGM
    tenk_data = CGMLoader().get_data(study_ids=['10K'])
    iglu_df = pd.read_csv('/net/mraid08/export/genie/LabData/Analyses/ayyak/CGM/iglu/iglu_days_2_13.csv', index_col=0)
    iglu_df['RegistrationCode'] = [s.split('/')[0] for s in iglu_df['id']]
    iglu_df['ConnectionID'] = [s.split('/')[-1] for s in iglu_df['id']]
    iglu_df = iglu_df.groupby('RegistrationCode').first().reset_index()
    iglu_df = iglu_df.set_index(['RegistrationCode', 'ConnectionID']).drop('id', axis=1)
    tenk_data.df_metadata = tenk_data.df_metadata.reindex(iglu_df.index).dropna(how='all')
    tenk_data.df_metadata['Date'] = pd.to_datetime([d.date() for d in pd.to_datetime(tenk_data.df_metadata['Period_start'])])
    tenk_data.df_metadata = tenk_data.df_metadata.reset_index().set_index(['RegistrationCode', 'Date'])
    tenk_data.df = iglu_df
    tenk_data.df.index = tenk_data.df_metadata.index
    clean_and_save(tenk_data, 'cgm')

    # Diet - short food names
    dl = DietLoggingLoader()
    tenk_data = dl.get_data(study_ids=['10K'])
    tenk_data.df = get_diet_logging_around_stage(tenk_data.df, stage='baseline', delta_before=2, delta_after=14)
    tenk_data = dl.daily_mean_food_consumption(df=tenk_data.df, kcal_limit=500, min_col_present_frac=0.05,
                                               level='shortname_eng')

    tenk_data.df = tenk_data.df.astype(float)
    # tenk_data.df = tenk_data.df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    tenk_data.df.replace({0: np.nan}, inplace=True)
    tenk_data.df = Loader._norm_dist_capping(tenk_data.df, {'sample_size_frac': 0.99, 'clip_sigmas': 10})
    tenk_data.df.fillna(0, inplace=True)
    tenk_data.df_metadata = tenk_data.df_metadata.reindex(tenk_data.df.index)
    tenk_data = add_date_from_body_measures(tenk_data, body_measures)
    clean_and_save(tenk_data, 'diet')

    # Diet full - short food names
    dl = DietLoggingLoader()
    tenk_data = dl.get_data(study_ids=['10K'])
    tenk_data.df = get_diet_logging_around_stage(tenk_data.df, stage='baseline', delta_before=2, delta_after=14)
    tenk_data = dl.daily_mean_food_consumption(df=tenk_data.df, kcal_limit=500, min_col_present_frac=0,
                                               level='shortname_eng')

    tenk_data.df = tenk_data.df.astype(float)
    # tenk_data.df = tenk_data.df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    tenk_data.df.replace({0: np.nan}, inplace=True)
    tenk_data.df = Loader._norm_dist_capping(tenk_data.df, {'sample_size_frac': 0.99, 'clip_sigmas': 10})
    tenk_data.df.fillna(0, inplace=True)
    tenk_data.df_metadata = tenk_data.df_metadata.reindex(tenk_data.df.index)
    tenk_data = add_date_from_body_measures(tenk_data, body_measures)
    clean_and_save(tenk_data, 'diet_full')

    # medications
    tenk_data = get_baseline_medications()
    tenk_data = Medications10KLoader().get_data(df=tenk_data.df, pivot_by=4)
    # in case there are multiple reportings, keep a positive one if exists
    tenk_data.df = tenk_data.df.groupby('RegistrationCode').max()
    medication_names = tenk_data.df.columns
    # medication matrix should be complete, adding participants from body measures loader
    # merging loaders to get metadata
    tenk_data = DataMerger([body_measures, tenk_data]).get_x(res_index_names=['RegistrationCode'])
    # keeping only the medication columns
    tenk_data.df = tenk_data.df.loc[:, medication_names]
    tenk_data.df.fillna(0, inplace=True)
    tenk_data.df = tenk_data.df.loc[:, tenk_data.df.sum() >= 10].astype(float)
    tenk_data.df = tenk_data.df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    tenk_data = add_date_from_body_measures(tenk_data, body_measures)
    clean_and_save(tenk_data, 'medications')

    # medical conditions
    tenk_data = get_baseline_medical_conditions()
    tenk_data.df = tenk_data.df.fillna(-1).pivot_table(index=['RegistrationCode'], columns=['medical_condition'],
                                                       values='Start')
    tenk_data.df_metadata = tenk_data.df_metadata.groupby('RegistrationCode').first()
    medical_condition_names = tenk_data.df.columns
    # merging loaders to get metadata
    tenk_data = DataMerger([body_measures, tenk_data]).get_x(res_index_names=['RegistrationCode'])
    # keeping only the medication columns
    tenk_data.df = tenk_data.df.loc[:, medical_condition_names]

    tenk_data.df.fillna(0, inplace=True)
    tenk_data.df.replace({-1: np.nan}, inplace=True)
    tenk_data.df = tenk_data.df.loc[:, tenk_data.df.sum() >= 10].astype(float)
    tenk_data = add_date_from_body_measures(tenk_data, body_measures)
    clean_and_save(tenk_data, 'medical_conditions')
    return


def run_RSCV_old(x_name, y_name=None, subsample_X=1, random_seed=0):
    if y_name is None:
        y_name = x_name
    print(x_name)
    output_path = os.path.join(pred_dir, '%s-age/' % x_name)
    subsample_name = ''
    if subsample_X < 1:
        subsample_name = 'subsample_X'
        mkdirifnotexists(os.path.join(pred_dir, subsample_name))
        output_path = os.path.join(pred_dir, subsample_name,
                                   '%s_%0.2f_%d-age/' % (x_name, subsample_X, random_seed))
    os_sys_RandomizedSearchCV(output_path=output_path,
                              path_to_X=os.path.join(Xs_dir, '%s.csv' % x_name),
                              path_to_Y=os.path.join(Ys_dir, '%s.csv' % y_name),
                              k_folds=5,
                              subsample_X=subsample_X,
                              random_seed=random_seed)
    # os.system(' '.join(['python', '~/PycharmProjects/LabData/LabData/DataPredictors/RandomizedSearchCV.py',
    #                     output_path,
    #                     '-path_to_X', os.path.join(Xs_dir, '%s.csv' % x_name),
    #                     '-path_to_Y', os.path.join(Ys_dir, '%s.csv' % y_name),
    #                     '-k_folds', '5', '-subsample_X', str(subsample_X), '-random_seed', str(random_seed)]))
    return

def run_age_predictions_old(only_per_sex=False, loaders=None, subsample_X=None): # TODO: update to work with new prediction code
    x_names = [s.split('.csv')[0] for s in os.listdir(Xs_dir) if '.csv' in s]

    if subsample_X is not None:
        if not (isinstance(subsample_X, list) or isinstance(subsample_X, tuple)):
            subsample_X = [subsample_X]

    print(x_names)

    config = q_setup()

    with config.qp(jobname='age_pred', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=200,
                   _mem_def='25G') as q:
        q.startpermanentrun()
        waiton = []
        for x_name in x_names:
            y_name = None
            if only_per_sex:
                if 'male' not in x_name:
                    continue
                y_name = x_name.split('_female')[0].split('_male')[0]
            if loaders is not None and (x_name not in loaders and y_name not in loaders):
                continue
            print(x_name, y_name)
            if subsample_X is not None:
                for seed, subsample in enumerate(subsample_X):
                    waiton.append(q.method(run_RSCV_old, (x_name, y_name, subsample, seed, )))
            else:
                waiton.append(q.method(run_RSCV_old, (x_name, y_name, )))
        res = q.waitforresults(waiton)


def run_age_predictions(only_per_sex=False, loaders=None):
    x_names = [s.split('.csv')[0] for s in os.listdir(Xs_dir) if '.csv' in s]

    config = q_setup()

    with config.qp(jobname='age_pred', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=200,
                   _mem_def='25G') as q:
        q.startpermanentrun()
        waiton = []
        for x_name in x_names:
            y_name = None
            if only_per_sex and 'male' not in x_name:
                continue
            if not only_per_sex and 'male' in x_name:
                continue
            y_name = x_name.split('_female')[0].split('_male')[0]
            if loaders is not None and (x_name not in loaders and y_name not in loaders):
                continue
            print(x_name, y_name)
            # waiton.append(q.method(run_RSCV, (x_name, y_name,)))
            if y_name is None:
                y_name = x_name
            print(x_name)
            output_path = os.path.join(pred_dir, '%s-age/' % x_name)
            x_path = os.path.join(Xs_dir, '%s.csv' % x_name)
            y_path = os.path.join(Ys_dir, '%s.csv' % y_name)

            waiton.append(q.method(run_RSCV, (output_path, x_path, y_path,)))
        res = q.waitforresults(waiton)


def run_RSCV(output_path, x_path, y_path):

    conf_path = create_pred_conf_file(output_path,
                                      x_path,
                                      y_path,
                                      **rscv_kwargs)

    os.system(' '.join(['python', '~/PycharmProjects/LabData/LabData/DataPredictors/RandomizedSearchCV.py',
                        '--conf', conf_path]))
    return

def create_pred_conf_file(output_path, x_path, y_path, **kwargs):
    conf_dir = mkdirifnotexists(os.path.join(pred_dir, 'prediction_configs'))
    filename = '%s-%s.conf' % (os.path.basename(x_path).split('.')[0], os.path.basename(y_path).split('.')[0])
    with open(os.path.join(conf_dir, filename), 'w') as h:
        h.write("output_dir = '%s'\n" % (output_path))
        h.write("path_to_X = '%s'\n" % (x_path))
        h.write("path_to_Y = '%s'\n" % (y_path))
        h.write("job_name = '%s'\n" % (filename.split('.conf')[0]))
        for k in kwargs:
            h.write("%s = '%s'\n" % (k, kwargs[k]))
        h.close()
    return os.path.join(conf_dir, filename)


def bootsrapping_r2(df, real_name, pred_name, n_bootstraps=1000):
    from sklearn.utils import resample
    r2_list = []
    for i in range(n_bootstraps):
        boot = resample(df.index, replace=True, n_samples=df.shape[0], random_state=i)
        r2_list.append(r2_score(df.loc[boot, real_name], df.loc[boot, pred_name]))
    return pd.Series(r2_list, name='r2')

def concat_results_and_compute_residuals(pred_col='age', bins=[40, 50, 60, 70]):
    r2_col = 'Coefficient_of_determination'
    n_col = 'Size'
    run_dirs = [d for d in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, d))]
    # xs = list(set([d.split('-')[0] for d in run_dirs]))
    # results = pd.DataFrame(columns=['full model all', 'full model all n',
    #                                 'full model male', 'full model male n',
    #                                 'full model female', 'full model female n',
    #                                 'male', 'male n',
    #                                 'female', 'female n'])
    results = pd.DataFrame()
    bs_est = pd.DataFrame()
    for run in run_dirs:
        print(run)
        if not os.path.exists(os.path.join(pred_dir, run, 'test_results.csv')):
            continue
        res = pd.read_csv(os.path.join(pred_dir, run, 'test_results.csv'), index_col=0)
        predictions_df = pd.read_csv(os.path.join(pred_dir, run, 'cv_predictions_df.csv'), index_col=0)\
            .rename(columns={pred_col: 'pred'})
        # bs = pd.read_pickle(os.path.join(pred_dir, run, 'bs_results.pkl'))['age']
        sex = 'all'
        x_name = run.split('-')[0]
        run_name = run.split('-')[0]
        if 'male' in run:
            sex = run.split('-')[0].split('_equal')[0].split('_')[-1]
            x_name = x_name.split('_%s' % sex)[0]
            run_name = run.split('-')[0].split(x_name + '_')[-1]
        x_df = pd.read_csv(os.path.join(Xs_dir, '%s.csv' % x_name), index_col=0)
        y_df = pd.read_csv(os.path.join(Ys_dir, '%s.csv' % x_name), index_col=0).rename(columns={pred_col: 'true'})
        df = y_df.join(predictions_df[['pred']]).dropna()
        df['residuals'] = df['pred'] - df['true']

        if sex == 'all':
            df = df.join(x_df[['gender']]).dropna()
            male_r2 = r2_score(y_true=df[df.gender == 1]['true'], y_pred=df[df.gender == 1]['pred'])
            female_r2 = r2_score(y_true=df[df.gender == 0]['true'], y_pred=df[df.gender == 0]['pred'])
            results.loc[x_name, 'full model all'] = res.loc[pred_col, r2_col]
            results.loc[x_name, 'full model all n'] = res.loc[pred_col, n_col]
            results.loc[x_name, 'full model male'] = male_r2
            results.loc[x_name, 'full model male n'] = df[df.gender == 1].shape[0]
            results.loc[x_name, 'full model female'] = female_r2
            results.loc[x_name, 'full model female n'] = df[df.gender == 0].shape[0]
        else:
            df['gender'] = 0 if 'female' in sex else 1
            results.loc[x_name, run_name] = res.loc[pred_col, r2_col]
            results.loc[x_name, '%s n' % run_name] = res.loc[pred_col, n_col]
        bs_res = bootsrapping_r2(df, real_name='true', pred_name='pred', n_bootstraps=1000)
        # results.loc[x_name, '%s 0.025' % run_name] = bs[r2_col].quantile(0.025)
        # results.loc[x_name, '%s 0.5' % run_name] = bs[r2_col].quantile(0.5)
        # results.loc[x_name, '%s 0.975' % run_name] = bs[r2_col].quantile(0.975)
        results.loc[x_name, '%s 0.025' % run_name] = bs_res.quantile(0.025)
        results.loc[x_name, '%s 0.5' % run_name] = bs_res.quantile(0.5)
        results.loc[x_name, '%s 0.975' % run_name] = bs_res.quantile(0.975)
        df['%s bin' % pred_col] = df['true'].astype(int)
        # compute ranks of the residuals binned by age and sex
        df['residuals rank'] = df[['residuals', 'gender', '%s bin' % pred_col]].groupby(['gender', '%s bin' % pred_col])\
            .apply(lambda x: x['residuals'].rank() / x.shape[0]).reset_index(['gender', '%s bin' % pred_col], drop=True)
        # compute STDs/Z-scores of the residuals binned by age and sex
        df['residuals zscore'] = df[['residuals', 'gender', '%s bin' % pred_col]].groupby(['gender', '%s bin' % pred_col]) \
            .apply(lambda x: (x['residuals'] - x['residuals'].mean()) / x['residuals'].std())\
            .reset_index(['gender', '%s bin' % pred_col], drop=True)
        df['residuals centered'] = df[['residuals', 'gender', '%s bin' % pred_col]].groupby(
            ['gender', '%s bin' % pred_col]) \
            .apply(lambda x: x['residuals'] - x['residuals'].mean()) \
            .reset_index(['gender', '%s bin' % pred_col], drop=True)
        df['ba (residuals centered)'] = df['%s bin' % pred_col] + df['residuals centered']
        df.to_csv(os.path.join(pred_dir, run, 'residuals.csv'))
        # continue

        lowess = sm.nonparametric.lowess
        xvals = np.arange(int(df['true'].min()), np.ceil(df['true'].max()), 0.2)
        yvals = lowess(df['pred'], df['true'], return_sorted=False, frac=0.33, xvals=xvals)
        # # plot true vs predicted
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.scatter(df['true'], df['pred'], alpha=0.7)
        ax.plot(xvals, yvals, color='black', linewidth=3)
        # sns.regplot(data=df, x='true', y='pred', ax=ax)
        # w = 200
        # ax.plot(df.sort_values('true')['true'],
        #         df.sort_values('true')['pred'].rolling(window=w, center=True, min_periods=10).mean(),
        #         linewidth=3, color='black', label='moving average')
        ax.set_xlabel('True %s' % pred_col, fontsize=15)
        ax.set_ylabel('Predicted %s' % pred_col, fontsize=15)
        ax.set_title(r'%s, %s, $R^2$=%0.2g' % (modality_name_mapping[x_name], sex,
                                               res.loc[pred_col, r2_col]), fontsize=15)
        ax.tick_params(labelsize=15)
        fig.tight_layout()
        plt.savefig(os.path.join(pred_dir, run, 'real_vs_pred.png'), dpi=200)
        plt.close()

        # compute the running pearson R
        running_r = compute_running_r2(df, bins=bins)
        running_r.to_csv(os.path.join(pred_dir, run, 'running_r.csv'))

        # temp_bs = bs[[r2_col]].copy()
        # temp_bs['run_name'] = run_name
        # temp_bs['x_name'] = x_name
        # bs_est = pd.concat((bs_est, temp_bs), axis=0)

    results.to_csv(os.path.join(pred_dir, 'results.csv'))
    # bs_est.to_csv(os.path.join(pred_dir, 'bs_est.csv'))
    return


def plot_r2_summary_barplot():
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    r2_res = res.loc[:, ~res.columns.str.contains(' n')]
    melted = r2_res.reset_index().melt(id_vars=['index']).rename(columns={'index': 'modality', 'value': 'r2',
                                                                          'variable': 'model'})
    sns.barplot(data=melted, x='modality', y='r2', hue='model', ax=ax)
    ax.set_xlabel('Data modality', fontsize=15)
    ax.set_ylabel('R2 of age', fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    plt.savefig(os.path.join(pred_dir, 'results.png'), dpi=200)

    # clean version
    r2_df = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)[['female_equal', 'male_equal']]
    r2_df = r2_df.sort_values('female_equal', ascending=False).rename(
        columns={'female_equal': 'Female', 'male_equal': 'Male'})
    r2_df = r2_df.reset_index().melt(id_vars='index')
    r2_df.replace({'index': modality_name_mapping}, inplace=True)
    # modality_name_mapping
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sns.barplot(data=r2_df, x='index', y='value', hue='variable', ax=ax)
    ax.set_xticklabels([s.get_text().replace('_', ' ') for s in ax.get_xticklabels()], rotation=45, ha='right')
    ax.set_xlabel('Data modality', fontsize=15)
    ax.set_ylabel(r'Chronological age $R^2$', fontsize=15)
    ax.tick_params(labelsize=15)
    ax.legend(loc='best', title='', fontsize=15)
    fig.tight_layout()
    plt.savefig(os.path.join(work_dir, 'results_by_gender.png'), dpi=200)

    # version with error bars
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    r = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0).filter(regex='equal')
    r = r.loc[~r.index.str.contains('full')]
    r1 = r.reset_index().melt(id_vars=['index'] + r.filter(regex='0.0|0.9').columns.tolist(),
                              value_vars=['female_equal 0.5'])
    r1['min_error'] = r1['value'] - r1['female_equal 0.025']
    r1['max_error'] = r1['female_equal 0.975'] - r1['value']
    r2 = r.reset_index().melt(id_vars=['index'] + r.filter(regex='0.0|0.9').columns.tolist(),
                              value_vars=['male_equal 0.5'])
    r2['min_error'] = r2['value'] - r2['male_equal 0.025']
    r2['max_error'] = r2['male_equal 0.975'] - r2['value']
    r = pd.concat((r1, r2), axis=0)
    r['max_r2'] = r[['female_equal 0.975', 'male_equal 0.975']].max(1)
    r = r.sort_values(['max_r2', 'index', 'variable'], ascending=False)
    r.replace({'index': modality_name_mapping}, inplace=True)
    r.replace({'variable': {'male_equal 0.5': 'Males', 'female_equal 0.5': 'Fales'}}, inplace=True)

    ax = sns.barplot(data=r, x='index', y='value', hue='variable', ax=ax)
    ax.set_xlabel('Data modality', fontsize=15)
    ax.set_ylabel(r'$R^2$ of chronological age', fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(loc='best', title='Sex')

    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords,
                yerr=r[["min_error", "max_error"]].iloc[pd.Series(x_coords).rank() - 1].T.values, fmt="none", c="k")
    fig.tight_layout()
    plt.savefig(os.path.join(work_dir, 'results_by_gender_w_errbars.png'), dpi=200)


def load_ranked_residuals(res, gender, permuted=False, rank_or_z='rank'):
    assert rank_or_z in ['rank', 'zscore', 'centered']
    residuals_dfs = []
    for x_name in res.index:
        if not os.path.exists(os.path.join(pred_dir, '%s%s-age' % (x_name, gender), 'residuals.csv')):
            continue
        temp_df = pd.read_csv(os.path.join(pred_dir, '%s%s-age' % (x_name, gender), 'residuals.csv'),
                              index_col=0)[['residuals %s' % rank_or_z]]\
            .rename(columns={'residuals %s' % rank_or_z: x_name})
        if permuted:
            temp_df[x_name] = np.random.permutation(temp_df[x_name])
        residuals_dfs.append(temp_df)
    residuals_df = pd.concat(residuals_dfs, axis=1)
    return residuals_df


def plot_true_vs_permuted_mean_residuals(res_r2_th=0, rank_or_z='rank'):
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    fig_dir = mkdirifnotexists(os.path.join(pred_residuals_fig_dir, 'true_vs_permuted_mean_%s_residuals' % rank_or_z))

    res[res['full model all'] > res_r2_th]

    for gender in ['', '_male', '_female', '_male_equal', '_female_equal']:
        residuals_df = load_ranked_residuals(res, gender, permuted=False, rank_or_z=rank_or_z)
        permuted_residuals_df = load_ranked_residuals(res, gender, permuted=True, rank_or_z=rank_or_z)

        mean_rank = residuals_df.loc[residuals_df.notnull().sum(1) >= 10].mean(1)
        mean_rank_p = permuted_residuals_df.loc[permuted_residuals_df.notnull().sum(1) >= 10].mean(1)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.hist([mean_rank, mean_rank_p], bins=15, label=['real', 'permuted %s' % rank_or_z])

        t, p = levene(mean_rank.dropna(), mean_rank_p.dropna())
        ax.legend(loc='best')
        ax.set_xlabel('Mean %s' % rank_or_z, fontsize=15)
        ax.set_ylabel('Number of participants', fontsize=15)
        ax.set_title('%s, p=%0.2g' % ('all' if gender == '' else gender[1:], p), fontsize=15)
        plt.savefig(os.path.join(fig_dir, 'real_vs_permuted_mean_%s%s.png' % (rank_or_z, gender)), dpi=200)


def plot_residuals_correlation_matrix(res_r2_th=0, rank_or_z='rank'):
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    res = res.loc[~res.index.str.contains('full')]
    fig_dir = mkdirifnotexists(os.path.join(pred_residuals_fig_dir, 'correlation_matrix'))

    res[res['full model all'] > res_r2_th]

    # for gender in ['', '_male', '_female', '_male_equal', '_female_equal']:
    #
    #     residuals_df = load_ranked_residuals(res, gender, permuted=False, rank_or_z=rank_or_z)
    #
    #     corr_mat = residuals_df.corr()
    #     mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    #     # corr_mat.iloc[range(corr_mat.shape[0]), range(corr_mat.shape[0])] = np.nan
    #     # correlate the correlation matrix
    #     cg1 = sns.clustermap(corr_mat, xticklabels=False, yticklabels=False, metric='correlation')
    #     plt.clf()
    #     corr_mat = corr_mat.iloc[cg1.dendrogram_row.reordered_ind, cg1.dendrogram_col.reordered_ind]
    #
    #     fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #     sns.heatmap(corr_mat, ax=ax, annot=True, mask=mask, center=0, fmt="0.2g", annot_kws={'size': 7})
    #     ax.set_title('Residual %s correlation heatmap%s' % (rank_or_z, ' '.join(gender.split('_'))), fontsize=15)
    #     fig.tight_layout()
    #     plt.savefig(os.path.join(fig_dir, 'residual_%s_corr_heatmap%s.png' % (rank_or_z, gender)), dpi=200)

    male_residuals_df = load_ranked_residuals(res, '_male_equal', permuted=False, rank_or_z=rank_or_z)
    female_residuals_df = load_ranked_residuals(res, '_female_equal', permuted=False, rank_or_z=rank_or_z)
    male_corr_mat = male_residuals_df.rename(columns=modality_name_mapping).corr()
    female_corr_mat = female_residuals_df.rename(columns=modality_name_mapping).corr()
    mask = np.triu(np.ones_like(male_corr_mat, dtype=bool), k=0)
    cg1 = sns.clustermap(male_corr_mat, xticklabels=False, yticklabels=False, metric='correlation')
    plt.clf()
    male_corr_mat = male_corr_mat.iloc[cg1.dendrogram_row.reordered_ind, cg1.dendrogram_col.reordered_ind].iloc[1:, :-1]
    female_corr_mat = female_corr_mat.iloc[cg1.dendrogram_row.reordered_ind, cg1.dendrogram_col.reordered_ind].iloc[1:, :-1]
    mask = np.triu(np.ones_like(male_corr_mat, dtype=bool), k=1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ax = axes[0]
    sns.heatmap(male_corr_mat, ax=ax, annot=True, mask=mask, center=0, fmt="0.2f", annot_kws={'size': 5}, cbar=False,
                cmap='RdBu_r', vmin=-0.2, vmax=0.5)
    ax.set_title('Males', fontsize=15)
    ax = axes[1]
    sns.heatmap(female_corr_mat, ax=ax, annot=True, mask=mask, center=0, fmt="0.2f", annot_kws={'size': 5}, cbar=False,
                cmap='RdBu_r', yticklabels=False, vmin=-0.2, vmax=0.5)
    ax.set_title('Females', fontsize=15)
    ax = axes[2]
    sns.heatmap(female_corr_mat - male_corr_mat, ax=ax, annot=True, mask=mask, center=0, fmt="0.2f",
                annot_kws={'size': 5}, cbar=False, cmap='RdBu_r', yticklabels=False, vmin=-0.2, vmax=0.5)
    ax.set_title('Females - Males', fontsize=15)
    clb = fig.colorbar(ax.get_children()[0], orientation="vertical",
                       ticks=[-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5], aspect=20)
    clb.ax.set_ylabel('Pearson correlation', fontsize=15)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'residual_%s_corr_heatmap_male_female_diff.png' % (rank_or_z)), dpi=200)

def plot_residuals_edges_clustermap(res_r2_th=0, q=0.03, rank_or_z='rank'):
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    fig_dir = mkdirifnotexists(os.path.join(pred_residuals_fig_dir, '%s_residuals_edges_clustermaps' % rank_or_z))
    res[res['full model all'] > res_r2_th]

    for gender in ['', '_male', '_female', '_male_equal', '_female_equal']:

        residuals_df = load_ranked_residuals(res, gender, permuted=False)
        mean_rank = residuals_df.loc[residuals_df.notnull().sum(1) >= 10].mean(1)
        groups = {'top': mean_rank[mean_rank >= mean_rank.quantile(1 - q)],
                  'bottom': mean_rank[mean_rank <= mean_rank.quantile(q)]}
        for gr in groups:
            residuals_gr = residuals_df.loc[groups[gr].index]
            mask = residuals_gr.isnull()
            # correlate the correlation matrix
            cg1 = sns.clustermap(residuals_gr.fillna(0.5), xticklabels=False, yticklabels=False, metric='euclidean')
            plt.clf()
            residuals_gr = residuals_gr.iloc[cg1.dendrogram_row.reordered_ind, cg1.dendrogram_col.reordered_ind]

            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            sns.heatmap(residuals_gr, ax=ax, mask=mask, center=0.5, cmap='RdBu', yticklabels=False)
            ax.set_title('Residual rank %s%0.1g%s' % (gr, q, ' '.join(gender.split('_'))), fontsize=15)
            # ax.set_yticks(None)
            fig.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'residual_rank_%s%0.1g%s.png' % (gr, q, gender)), dpi=200)


def tsne_and_counts_over_edges(res_r2_th=0, q=0.1, rank_or_z='rank'):
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    fig_dir = mkdirifnotexists(os.path.join(pred_residuals_fig_dir, 'tSNE'))

    res[res['full model all'] > res_r2_th]

    # screen some of the modalities to reduce collinearity
    use_cols = ['diet', 'blood_tests', 'medical_conditions', 'cgm', 'abi', 'dxa',
       'liver_ultrasound', 'serum_met', 'body_measures', 'microbiome',
       'carotid_ultrasound', 'itamar_sleep'] #  'medications', 'retina_scan'
    res = res.loc[use_cols]

    for gender in ['', '_male', '_female', '_male_equal', '_female_equal']:
        residuals_df = load_ranked_residuals(res, gender, permuted=False, rank_or_z=rank_or_z)
        permuted_residuals_df = load_ranked_residuals(res, gender, permuted=True, rank_or_z=rank_or_z)
        for res_df, permute in zip([residuals_df, permuted_residuals_df], [False, True]):
            res_df = res_df.loc[res_df.notnull().sum(1) >= 10]

            res_df_upper_q = res_df >= (1-q)
            res_df_upper_q[res_df.isnull()] = np.nan
            res_df_lower_q = res_df <= q
            res_df_lower_q[res_df.isnull()] = np.nan

            # run tSNE and save embedding
            embd = TSNE(n_components=2, learning_rate='auto', perplexity=20, metric='euclidean', init='random',
                        random_state=0).fit_transform(res_df.fillna(0.5))
            embd = umap.UMAP(random_state=0).fit_transform(res_df)
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.scatter(embd[:, 0], embd[:, 1], alpha=0.7)
            ax.tick_params(labelsize=15)
            ax.set_xlabel('tSNE 1', fontsize=15)
            ax.set_ylabel('tSNE 2', fontsize=15)
            ax.set_title('tSNE plot over %s residuals\n(%s)%s' %
                         (rank_or_z, ' '.join(gender.split('_')).strip(), ' permuted' if permute else ''), fontsize=15)
            fig.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'tSNE%s%s.png' % (gender, '_permuted' if permute else '')), dpi=200)

            for df, name in zip([res_df_upper_q, res_df_lower_q], ['top%0.1g' % q, 'bottom%0.1g' % q]):


                # count combinations and show 10 most common
                # c = res_df_lower_q.dropna().apply(lambda x: ','.join(set(x[x==1].index)), axis=1).value_counts()
                pass

                # compute statistic for random combinations and attach

                # alternatively, permute the columns and perform the counting

def save_residuals(rank_or_z='zscore'):
    cat_to_exclude = ['diet_questions', 'medical_conditions', 'medications']

    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0).drop(cat_to_exclude, axis=0)
    res = res.loc[~res.index.str.contains('full')]

    male_residuals_df = load_ranked_residuals(res, '_male_equal', permuted=False, rank_or_z=rank_or_z).rename(
        columns=modality_name_mapping)
    female_residuals_df = load_ranked_residuals(res, '_female_equal', permuted=False, rank_or_z=rank_or_z).rename(
        columns=modality_name_mapping)

    male_residuals_df['gender'] = 1
    female_residuals_df['gender'] = 0

    # think whetehr to add age as covariate

    df = pd.concat((male_residuals_df, female_residuals_df), axis=0)

    df.to_csv(os.path.join(pred_dir, 'residuals_df_equal%s.csv' % rank_or_z))
    return


def correlate_residuals_with_medical_conditions(rank_or_z='rank'):
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    fig_dir = mkdirifnotexists(os.path.join(pred_residuals_fig_dir, 'medical_conditions_corrs'))
    boxplots_dir = mkdirifnotexists(os.path.join(fig_dir, 'boxplots'))
    # res[res['full model all'] > res_r2_th]

    # screen some of the modalities to reduce collinearity
    # use_cols = ['diet', 'blood_tests', 'cgm', 'abi', 'dxa',
    #             'liver_ultrasound', 'serum_met', 'body_measures', 'microbiome',
    #             'carotid_ultrasound', 'itamar_sleep', 'retina_scan']  # 'medications', 'retina_scan'
    # use_cols = ['blood_lipids', 'body_composition', 'bone_density', 'diet', 'frailty',
                # 'glycemic_status', 'liver', 'microbiome', 'sleep', 'vascular_system', 'immune_system', 'lifestyle']
    use_cols = ['blood_lipids', 'body_composition', 'bone_density', 'diet', 'frailty', 'glycemic_status', 'liver',
                'microbiome', 'sleep', 'cardiovascular_system', 'immune_system', 'lifestyle', 'hematopoietic_system',
                'renal_function']

    res = res.loc[use_cols]

    medical_conditions = pd.read_csv(os.path.join(Xs_dir, 'medical_conditions.csv'), index_col=0).drop('gender', axis=1)
    medical_conditions = medical_conditions.rename(
        columns={k: k.replace('BlockL1-', '').replace('BlockL2-', '') for k in medical_conditions.columns})
    medical_conditions_cols = get_baseline_medical_conditions().df_columns_metadata[['english_name', 'ICD11Code']]
    medical_conditions_cols = medical_conditions_cols.set_index('ICD11Code').to_dict()
    medical_conditions = medical_conditions.rename(columns=medical_conditions_cols['english_name'])
    medical_conditions.T.loc[:, []].to_csv(os.path.join(fig_dir, 'medical_conditions_names.csv'))

    medical_condition_groups = pd.read_csv(os.path.join(work_dir, 'medical_conditions_names_grouped_by_Lee.csv'),
                                           header=None).dropna(how='all')
    for med_c in medical_condition_groups.iloc[:, 0].dropna():
        cols = medical_condition_groups.loc[medical_condition_groups.iloc[:, 0]==med_c].iloc[0, 1:].dropna()
        medical_conditions[med_c] = medical_conditions.loc[:, cols].any(1).astype(float)
        medical_conditions.drop(cols, axis=1, inplace=True)
    print(medical_conditions.shape)
    # return

    spearman_df = pd.DataFrame(index=use_cols, columns=medical_conditions.columns)
    numbers = pd.DataFrame(columns=['system', 'medical diagnosis', 'variable', 'value'])
                                    # 'male (no)', 'male (yes)', 'female (no)', 'female (yes)'])

    for gender in ['_male_equal', '_female_equal']:  # '', '_male', '_female',
        logit_df = pd.DataFrame(index=use_cols, columns=medical_conditions.columns)
        logit_df_p = pd.DataFrame(index=use_cols, columns=medical_conditions.columns)

        residuals_df = load_ranked_residuals(res, gender, permuted=False, rank_or_z=rank_or_z)
        # residuals_df.to_csv(os.path.join(fig_dir, 'residuals_df%s_%s.csv' % (gender, rank_or_z)))
        # continue
        residuals_df = residuals_df.join(medical_conditions, how='left')
        residuals_df['const'] = 1
        for x_name in use_cols:
            print(x_name)
            temp_df = residuals_df.dropna(subset=[x_name]).copy()
            temp_df = temp_df[[x_name, 'const']].join(temp_df.loc[:, medical_conditions.columns]
                                             .loc[:, temp_df[medical_conditions.columns].sum() > temp_df.shape[0] * 0.01])
            print(temp_df.shape)
            for med in temp_df.columns:
                if med in [x_name, 'const']:
                    continue
                numbers.loc[numbers.shape[0]] = [modality_name_mapping[x_name], med, '%s (yes)' % gender.split('_')[1],
                                                 temp_df[med].sum()]
                numbers.loc[numbers.shape[0]] = [modality_name_mapping[x_name], med, '%s (no)' % gender.split('_')[1],
                                                 temp_df[med].dropna().shape[0] - temp_df[med].sum()]

                try:
                    log = Logit(temp_df[med].astype(float),
                                temp_df[[x_name, 'const']], missing='drop').fit(disp=0)
                    logit_df.loc[x_name, med] = np.exp(log.params.loc[x_name])
                    logit_df_p.loc[x_name, med] = log.pvalues.loc[x_name]
                except:
                    logit_df.loc[x_name, med], logit_df_p.loc[x_name, med] = np.nan, np.nan
                # try:
                #     t, p = mannwhitneyu(residuals_df[residuals_df[med] == 1][x_name].dropna(),
                #                         residuals_df[residuals_df[med] == 0][x_name].dropna())
                # except:
                #     t, p = 0, 0.5

                # if p < 0.001:
                #     fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                #     sns.boxplot(data=residuals_df, x=med, y=x_name, ax=ax)
                #     ax.set_xlabel(med, fontsize=15)
                #     ax.set_ylabel('%s %s' % (x_name, rank_or_z), fontsize=15)
                #     ax.set_title('MWU P=%0.1g' % p)
                #     fig.tight_layout()
                #     plt.savefig(os.path.join(boxplots_dir, '%s-%s%s.png' % (x_name, med, gender)), dpi=200)

                # r, p = spearmanr(residuals_df[x_name], residuals_df[med], nan_policy='omit')
                # spearman_df.loc[x_name, med] = (r, p)


        # logit_df.to_csv(os.path.join(fig_dir, 'logit_df%s_%s.csv' % (gender, rank_or_z)))
        # logit_df_p.to_csv(os.path.join(fig_dir, 'logit_df_p%s_%s.csv' % (gender, rank_or_z)))
        logit_df = logit_df.T.rename(columns=modality_name_mapping).T
        logit_df_p = logit_df_p.T.rename(columns=modality_name_mapping).T

        logit_df.to_csv(os.path.join(fig_dir, 'logit_df%s_%s_full.csv' % (gender, rank_or_z)))
        logit_df_p.to_csv(os.path.join(fig_dir, 'logit_df_p%s_%s_full.csv' % (gender, rank_or_z)))

        # mask = logit_df_p > 1e-4

        mask = pd.DataFrame(~fdr_correction(logit_df_p.stack(), alpha=0.01)[0], index=logit_df_p.stack().index).unstack()
        mask.columns = [m[-1] for m in mask.columns]
        mask = mask.reindex(logit_df_p.columns, axis=1)
        mask.fillna(True, inplace=True)
        # mask = pd.DataFrame(~fdr_correction(logit_df_p, alpha=0.01)[0], index=logit_df_p.index, columns=logit_df_p.columns)
        logit_df[mask] = np.nan
        logit_df.dropna(how='all', axis=1, inplace=True)
        logit_df.dropna(how='all', axis=0, inplace=True)
        logit_df.fillna(0, inplace=True)
        logit_df_p = logit_df_p.loc[logit_df.index, logit_df.columns]
        # mask = logit_df_p > 1e-4
        mask = mask.loc[logit_df_p.index, logit_df_p.columns]
        # correlate the correlation matrix
        cg1 = sns.clustermap(logit_df, xticklabels=False, yticklabels=False, metric='euclidean')
        plt.clf()
        logit_df = logit_df.iloc[cg1.dendrogram_row.reordered_ind, cg1.dendrogram_col.reordered_ind]
        mask = mask.iloc[cg1.dendrogram_row.reordered_ind, cg1.dendrogram_col.reordered_ind]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.heatmap(logit_df, ax=ax, mask=mask, center=1., cmap='RdBu', xticklabels=True, cbar=False)

        clb = fig.colorbar(ax.get_children()[0], orientation="vertical", aspect=20)
        clb.ax.set_ylabel('Odds Ratio', fontsize=15)
        # ax.tick_params(labelsize=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Residuals %s - medical conditions %s' % (rank_or_z, ' '.join(gender.split('_'))), fontsize=15)
        fig.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'logit_df%s_%s.png' % (gender, rank_or_z)), dpi=200, bbox_inches='tight')
        logit_df.to_csv(os.path.join(fig_dir, 'logit_df%s_%s.csv' % (gender, rank_or_z)))

    numbers.pivot_table(index=['system', 'medical diagnosis'], columns=['variable'], values='value')\
        .to_csv(os.path.join(fig_dir, 'numbers_%s.csv' % (rank_or_z)))

        # spearman_df.to_csv(os.path.join(fig_dir, 'spearman%s_%s.csv' % (gender, rank_or_z)))
#

def compute_running_r2(df, bins):
    df['bin'] = pd.cut(df['true'], bins=bins)
    # return df.groupby('bin').apply(lambda x: r2_score(y_true=x['true'], y_pred=x['pred']))
    return pd.concat((df.groupby('bin').apply(lambda x: pearsonr(x['true'], x['pred'])[0]).rename('r'),
                      df.groupby('bin').apply(lambda x: x.shape[0]).rename('n')), axis=1)


def plot_running_scores():
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    for x_name in res.index:
        both_scores = pd.read_csv(os.path.join(pred_dir, '%s-age' % x_name, 'running_r.csv'),
                                  index_col=0).rename(columns={'r': 'all', 'n': 'n all'})
        male_scores = pd.read_csv(os.path.join(pred_dir, '%s_male_equal-age' % x_name, 'running_r.csv'),
                                  index_col=0).rename(columns={'r': 'male', 'n': 'n male'})
        female_scores = pd.read_csv(os.path.join(pred_dir, '%s_female_equal-age' % x_name, 'running_r.csv'),
                                    index_col=0).rename(columns={'r': 'female', 'n': 'n female'})

        scores = both_scores.join(male_scores).join(female_scores)
        melted_r = scores.reset_index().melt(id_vars=['bin'], value_vars=['all', 'male', 'female'])\
            .rename(columns={'bin': 'Age bin', 'value': 'Pearson r', 'variable': 'Group'})
        melted_n = scores.reset_index().melt(id_vars=['bin'], value_vars=['n all', 'n male', 'n female']) \
            .rename(columns={'bin': 'Age bin', 'value': 'Group size', 'variable': 'Group'})

        ax_dict = plt.figure(constrained_layout=True, figsize=(7, 5.5)).subplot_mosaic(
            """
            A
            B
            B
            """,
            gridspec_kw={"wspace": 0, "hspace": 0},
        )
        ax_r = ax_dict["B"]
        sns.pointplot(data=melted_r, x='Age bin', y='Pearson r', hue='Group', ax=ax_r)
        ax_r.set_ylabel("Real vs predicted Pearson R", fontsize=15)
        ax_r.set_xlabel('Age bin', fontsize=15)
        ax_r.tick_params(labelsize=15)
        ax_r.legend(loc='best', fontsize=15)

        ax_n = ax_dict["A"]
        sns.barplot(data=melted_n, x='Age bin', y='Group size', hue='Group', ax=ax_n)
        ax_n.set_ylabel("Group size", fontsize=15)
        ax_n.legend().set_visible(False)
        ax_n.set_xlabel('')
        ax_n.set_title(x_name, fontsize=15)
        ax_n.tick_params(labelsize=15)
        # ax_n.set_xticklabels(None)
        plt.tight_layout()
        plt.savefig(os.path.join(pred_dir, '%s-age' % x_name, 'running_r.png'), dpi=200)


def compare_shap_values():
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    for x_name in res.index:
        male_shaps = pd.read_csv(os.path.join(pred_dir, '%s_male_equal-age' % x_name, 'abs_signed_shap.csv'),
                                 index_col=0).loc['age'].rename('male')
        female_shaps = pd.read_csv(os.path.join(pred_dir, '%s_female_equal-age' % x_name, 'abs_signed_shap.csv'),
                                 index_col=0).loc['age'].rename('female')
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.scatter(male_shaps, female_shaps, alpha=0.5)
        ax.set_xlabel('Male SHAP values', fontsize=15)
        ax.set_ylabel('Female SHAP values', fontsize=15)
        ax.set_title(x_name)
        xpoints = ypoints = ax.set_xlim()
        ax.plot(xpoints, ypoints, linestyle='--', color='red', scalex=False, scaley=False)
        plt.tight_layout()
        plt.savefig(os.path.join(pred_dir, '%s-age' % x_name, 'shap_male_vs_female.png'), dpi=200)

        pd.concat((male_shaps, female_shaps, (female_shaps - male_shaps).abs().rename('diff')), axis=1)\
            .sort_values('diff').to_csv(os.path.join(pred_dir, '%s-age' % x_name, 'shap_male_vs_female.csv'))


def compare_shap_values_subsample(subsample_dir):
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    for gender in ['male', 'female']:
        for x_name in res.index:
            shaps1 = pd.read_csv(os.path.join(pred_dir, subsample_dir, '%s_%s_equal_%s-age' % (x_name, gender, '0.50_0'), 'abs_signed_shap.csv'),
                                     index_col=0).loc['age'].rename('iter 1')
            shaps2 = pd.read_csv(os.path.join(pred_dir, subsample_dir, '%s_%s_equal_%s-age' % (x_name, gender, '0.50_1'), 'abs_signed_shap.csv'),
                                     index_col=0).loc['age'].rename('iter 2')
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.scatter(shaps1, shaps2, alpha=0.5)
            ax.set_xlabel('Iter 1 SHAP values', fontsize=15)
            ax.set_ylabel('Iter 2 SHAP values', fontsize=15)
            ax.set_title(x_name)
            xpoints = ypoints = ax.set_xlim()
            ax.plot(xpoints, ypoints, linestyle='--', color='red', scalex=False, scaley=False)
            plt.tight_layout()
            plt.savefig(os.path.join(pred_dir, subsample_dir, '%s_%s_equal_%s-age' % (x_name, gender, '0.50_0'), 'shap_iter1_vs_iter2.png'), dpi=200)

            pd.concat((shaps1, shaps2, (shaps2 - shaps1).abs().rename('diff')), axis=1)\
                .sort_values('diff').to_csv(os.path.join(pred_dir, subsample_dir, '%s_%s_equal_%s-age' % (x_name, gender, '0.50_0'), 'shap_iter1_vs_iter2.csv'))


def compute_single_mantel_test(x1_path, x2_path, name, distance_metric='euclidean', perms=10000, mantel_method='pearson',
                               tail='upper'):
    print(name)
    x1 = pd.read_csv(x1_path, index_col=0).astype(float)
    x2 = pd.read_csv(x2_path, index_col=0).astype(float)
    intersecting_indeces = x1.index.intersection(x2.index)
    if len(intersecting_indeces) > 5000:
        np.random.seed(0)
        intersecting_indeces = np.random.choice(intersecting_indeces, 5000, replace=False)
    x1 = x1.loc[intersecting_indeces]
    x2 = x2.loc[intersecting_indeces]

    if x1.shape[0] == 0:
        print('There are no overlapping samples')
        return

    if distance_metric in ['spearman', 'pearson']:
        x1_dist = 1 - x1.T.corr(distance_metric)
        x2_dist = 1 - x2.T.corr(distance_metric)

    elif distance_metric == 'euclidean':
        x1 = (x1 - x1.mean()) / x1.std()
        x2 = (x2 - x2.mean()) / x2.std()
        x1_dist = nan_euclidean_distances(x1)
        x2_dist = nan_euclidean_distances(x2)
        x1_dist = np.maximum(x1_dist, x1_dist.T)
        x2_dist = np.maximum(x2_dist, x2_dist.T)
        x1_dist = pd.DataFrame(x1_dist, index=x1.index, columns=x1.index)
        x2_dist = pd.DataFrame(x2_dist, index=x2.index, columns=x2.index)

    else:
        raise
    x1_dist = greedy_drop_na(x1_dist)
    x2_dist = greedy_drop_na(x2_dist)
    intersecting_indeces = x1_dist.index.intersection(x2_dist.index)
    x1_dist = x1_dist.loc[intersecting_indeces, intersecting_indeces]
    x2_dist = x2_dist.loc[intersecting_indeces, intersecting_indeces]
    size = x1_dist.shape[0]
    x1_dist = squareform(x1_dist)
    x2_dist = squareform(x2_dist)

    mt = mantel.test(x1_dist, x2_dist, perms=perms, method=mantel_method, tail=tail)

    results = pd.DataFrame(columns=['r', 'p', 'z', 'method', 'perms', 'mean', 'std', 'tail', 'size', 'distance_metric'])
    results.loc[name] = [mt.r, mt.p, mt.z, mt.method, mt.perms, mt.mean, mt.std, mt.tail, size, distance_metric]
    if os.path.exists(os.path.join(mantel_dir, '%s.csv' % name)):
        old_results = pd.read_csv(os.path.join(mantel_dir, '%s.csv' % name), index_col=0)
        results = old_results.append(results)
    results.to_csv(os.path.join(mantel_dir, '%s.csv' % name))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist(mt.correlations[1:], bins=30, label='random')
    ax.axvline(x=mt.r, label='true', color='black', linestyle='--')
    ax.set_xlabel('Correlation coefficient', fontsize=15)
    ax.set_ylabel('Number of permutations', fontsize=15)
    ax.set_title(name, fontsize=15)
    ax.tick_params(labelsize=15)
    ax.legend(loc='upper left', fontsize=15)
    fig.tight_layout()
    plt.savefig(os.path.join(mantel_dir, '%s_%s.png' % (name, distance_metric)), dpi=200)


def greedy_drop_na(df):
    while df.isnull().sum().sum() > 0:
        na_count = df.isnull().sum().sort_values(ascending=False)
        df.drop(na_count.index[0], axis=0, inplace=True)
        df.drop(na_count.index[0], axis=1, inplace=True)
    return df


def compute_all_mantel_tests(distance_metric='euclidean'):
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    x_names = res.index

    config = q_setup()

    with config.qp(jobname='mantel', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=200,
                   _mem_def='25G') as q:
        q.startpermanentrun()
        waiton = []
        for x_name1, x_name2 in combinations_with_replacement(x_names, 2):
            # keep a lexicography ordering of the loaders
            if x_name1 == x_name2:
                continue
            print(x_name1, x_name2)
            for gender in ['', '_male', '_female']:
                x1_path = os.path.join(Xs_dir, '%s%s.csv' % (x_name1, gender))
                x2_path = os.path.join(Xs_dir, '%s%s.csv' % (x_name2, gender))
                name = '%s-%s%s' % (x_name1, x_name2, gender)
                waiton.append(q.method(compute_single_mantel_test, (x1_path, x2_path, name, distance_metric, )))
        res = q.waitforresults(waiton)


def bootstrap_sex_spec_corrs():
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    x_names = res.index

    config = q_setup()

    with config.qp(jobname='boot', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=200,
                   _mem_def='25G') as q:
        q.startpermanentrun()
        waiton = []
        for x_name in x_names:
            for gender in ['male', 'female']:

                waiton.append(q.method(bootstrap_sex_spec_corrs_func, (x_name, gender,)))
        res = q.waitforresults(waiton)


def bootstrap_sex_spec_corrs_func(x_name, gender):
    print(x_name, gender)
    n_bootstraps = 1000
    mkdirifnotexists(os.path.join(sex_spec_corrs_dir, x_name))
    x = pd.read_csv(os.path.join(Xs_dir, '%s_%s_equal.csv' % (x_name, gender)), index_col=0).astype(float)
    y = pd.read_csv(os.path.join(Ys_dir, '%s.csv' % x_name), index_col=0).astype(float).iloc[:, 0]

    i = x.index.intersection(y.index)
    x = x.loc[i]
    y = y.loc[i]

    rs = x.apply(lambda xi: spearmanr(xi, y, nan_policy='omit')[0])
    rs.to_csv(os.path.join(sex_spec_corrs_dir, x_name, 'rs_%s.csv' % gender))

    bootstrap_df = []
    for i in range(n_bootstraps):
        np.random.seed(i)
        bs_i = np.random.choice(y.index, size=y.shape[0], replace=True)
        y_i = y.loc[bs_i]
        bootstrap_df.append(x.loc[bs_i].apply(lambda xi: spearmanr(xi, y_i, nan_policy='omit')[0]))
        if i % 10 == 0:
            print(i)
    bootstrap_df = pd.concat(bootstrap_df, axis=1).T
    bootstrap_df.to_csv(os.path.join(sex_spec_corrs_dir, x_name, 'bootstrap_df_%s.csv' % gender))

def analyze_sex_spec_corrs():
    res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
    x_names = res.index

    for x_name in x_names:
        rs_m = pd.read_csv(os.path.join(sex_spec_corrs_dir, x_name, 'rs_male.csv'), index_col=0)['0']
        rs_f = pd.read_csv(os.path.join(sex_spec_corrs_dir, x_name, 'rs_female.csv'), index_col=0)['0']

        try:
            bs_m = pd.read_csv(os.path.join(sex_spec_corrs_dir, x_name, 'bootstrap_df_male.csv'), index_col=0)
            bs_f = pd.read_csv(os.path.join(sex_spec_corrs_dir, x_name, 'bootstrap_df_female.csv'), index_col=0)
        except:
            continue

        m_greater = rs_m > rs_f
        f_greater = rs_m < rs_f

        results = pd.DataFrame(index=rs_m.index, columns=['male r', 'female r', 'male r mean', 'male r std',
                                                          'female r mean', 'female r std', 'p', 'p hard'])
        results.loc[:, 'male r'] = rs_m.values
        results.loc[:, 'female r'] = rs_f.values
        results.loc[:, 'male r mean'] = bs_m.mean()
        results.loc[:, 'female r mean'] = bs_f.mean()
        results.loc[:, 'male r std'] = bs_m.std()
        results.loc[:, 'female r std'] = bs_f.std()
        results.loc[m_greater, 'p'] = 1 - (rs_m.loc[m_greater] > bs_f.loc[:, m_greater]).sum() / bs_f.shape[0]
        results.loc[f_greater, 'p'] = 1 - (rs_f.loc[f_greater] > bs_m.loc[:, f_greater]).sum() / bs_m.shape[0]
        results.loc[:, 'p'] = results.loc[:, 'p'].apply(lambda x: max(1./bs_m.shape[0], x))
        results.loc[:, 'p hard'] = (bs_m.min() > bs_f.max()) | (bs_m.max() < bs_f.min())

        results.to_csv(os.path.join(sex_spec_corrs_dir, x_name, 'results.csv'))
        for hard_exp in results[results['p hard']].index:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.hist([bs_m[hard_exp], bs_f[hard_exp]], bins=10, label=['Male', 'Female'])
            ax.legend(fontsize=15)
            ax.tick_params(labelsize=15)
            ax.set_xlabel('Spearman corr with age', fontsize=15)
            ax.set_ylabel('Number of bootstrap iterations', fontsize=15)
            ax.set_title(hard_exp, fontsize=15)
            fig.tight_layout()
            plt.savefig(os.path.join(sex_spec_corrs_dir, x_name, '%s.png' % hard_exp), dpi=200)


def loess_age_vs_metabs(x_name = 'serum_met'):
    loess_dir = mkdirifnotexists(os.path.join(work_dir, 'sex_specific_corrs', 'loess'))
    config = q_setup()

    with config.qp(jobname='loess', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=200,
                   _mem_def='25G') as q:
        q.startpermanentrun()
        waiton = []
        # x_name = 'serum_met'

        x_m = pd.read_csv(os.path.join(Xs_dir, '%s_%s_equal.csv' % (x_name, 'male')), index_col=0).astype(float)
        x_f = pd.read_csv(os.path.join(Xs_dir, '%s_%s_equal.csv' % (x_name, 'female')), index_col=0).astype(float)
        x = pd.concat((x_m, x_f), axis=0)
        x = x.apply(lambda x: (x-x.mean())/x.std())
        x_m, x_f = x.loc[x_m.index], x.loc[x_f.index]
        x_m.to_csv(os.path.join(loess_dir, '%s_%s_equal.csv' % (x_name, 'male')))
        x_f.to_csv(os.path.join(loess_dir, '%s_%s_equal.csv' % (x_name, 'female')))

        for gender in ['male', 'female']:
                waiton.append(q.method(loess_age_vs_metabs_func, (x_name, gender, 1./3, loess_dir, )))
        res = q.waitforresults(waiton)

def loess_age_vs_metabs_func(x_name, gender, frac, loess_dir):
    lowess = sm.nonparametric.lowess

    x = pd.read_csv(os.path.join(loess_dir, '%s_%s_equal.csv' % (x_name, gender)), index_col=0).astype(float)
    y = pd.read_csv(os.path.join(Ys_dir, '%s.csv' % x_name), index_col=0).astype(float).iloc[:, 0]

    i = x.index.intersection(y.index)
    x = x.loc[i]
    y = y.loc[i]

    xvals = np.arange(int(y.min()), np.ceil(y.max()), 0.2)
    loess_y = x.apply(lambda xi: lowess(xi, y, return_sorted=False, frac=frac, xvals=xvals))
    loess_y.index = xvals
    # adjust the mean to stay the same as in the original data
    loess_y = loess_y - loess_y.mean() + x.mean()
    loess_y.to_csv(os.path.join(loess_dir, '%s_%s_equal-loess_frac%0.2f.csv' % (x_name, gender, frac)))
    # loess_y = pd.read_csv(os.path.join(loess_dir, '%s_%s_equal-loess_frac%0.2f.csv' % (x_name, gender, frac)), index_col=0)

    # imputer = KNNImputer(n_neighbors=50)
    # loess_y = pd.DataFrame(imputer.fit_transform(loess_y), index=loess_y.index, columns=loess_y.columns)
    # loess_y.to_csv(os.path.join(loess_dir, '%s_%s_equal-loess_frac%0.2f_knn_imputed.csv' % (x_name, gender, frac)))

    y.to_csv(os.path.join(loess_dir, '%s_%s_equal-age.csv' % (x_name, gender)))
    return


def gender_stratified_lipid_trajectories(x_name = 'serum_met'):


    loess_dic = {}
    y_dic = {}

    loess_dir = os.path.join(work_dir, 'sex_specific_corrs', 'loess')
    frac = 1. / 3

    for gender in ['male', 'female']:
        x_loess = pd.read_csv(os.path.join(loess_dir, '%s_%s_equal-loess_frac%0.2f.csv' % (x_name, gender, frac)),
                              index_col=0).astype(float)
        y = pd.read_csv(os.path.join(loess_dir, '%s_%s_equal-age.csv' % (x_name, gender)), index_col=0)

        y_dic[gender] = y

        loess_dic[gender] = x_loess
    from scipy.cluster.hierarchy import cut_tree, linkage
    from scipy.spatial.distance import pdist

    gender = 'female'
    df_f = loess_dic['female'].T.dropna().astype(float).copy()
    df_m = loess_dic['male'].T.dropna().astype(float).copy()
    df = pd.concat((df_f, df_m), axis=1).dropna()

    df.columns = [np.round(c, 2) for c in df.columns]

    dist_df = pdist(df, metric='euclidean')
    linkage_df = linkage(dist_df, 'complete')
    labels_df = pd.Series(cut_tree(linkage_df, height=9).ravel(), index=df.index).sort_values()
    # labels_df = labels_df[labels_df>1]
    df = df.loc[labels_df.index]
    labels_df.value_counts()
    labels_df.to_csv(os.path.join(loess_dir, 'labels_df.csv'))

    df_clustered = df * np.nan
    for gr in labels_df.unique():
        if df.loc[labels_df == gr].shape[0] > 1:
            cg1 = sns.clustermap(df.loc[labels_df == gr], xticklabels=False, yticklabels=False, metric='euclidean',
                                 col_cluster=False)
            plt.clf()
            df_clustered.loc[labels_df == gr, :] = df.loc[labels_df == gr].iloc[cg1.dendrogram_row.reordered_ind,
                                                   :].values
        else:
            pass
            # df_clustered.loc[labels_df == gr, :] = df.loc[labels_df == gr]
    df_clustered.dropna(inplace=True)

    from matplotlib.gridspec import GridSpec
    import met_brewer

    # fig = plt.figure(figsize=(12, 14))
    fig = plt.figure(figsize=(10, 10))
    # gs = GridSpec(21, 20, hspace=0.05)
    gs = GridSpec(16, 20, hspace=0.05)

    ax_heatmap_f = fig.add_subplot(gs[0:5, 1:10])
    ax_heatmap_m = fig.add_subplot(gs[0:5, 10:19])
    ax_heatmap_colorbar = fig.add_subplot(gs[1:4, 19])
    ax_label = fig.add_subplot(gs[0:5, 0])

    sns.heatmap(df_clustered.iloc[:, :df_f.shape[1]], vmax=1, vmin=-1, center=0, cmap='RdBu_r',
                ax=ax_heatmap_f, yticklabels=False, cbar=False, xticklabels=True)
    sns.heatmap(df_clustered.iloc[:, df_f.shape[1]:], vmax=1, vmin=-1, center=0, cmap='RdBu_r',
                ax=ax_heatmap_m, yticklabels=False, cbar=False, xticklabels=True)
    clb = fig.colorbar(ax_heatmap_f.get_children()[0], cax=ax_heatmap_colorbar, orientation="vertical",
                       ticks=[-1, -0.5, 0, 0.5, 1], aspect=20)
    clb.ax.set_ylabel('z score', fontsize=15)
    ax_heatmap_f.set_xticklabels([int(c) if c in np.arange(40, 75, 5) else '' for c in df.columns[:df_f.shape[1]]],
                                 rotation=0, ha='center', fontsize=15)
    ax_heatmap_f.xaxis.set_ticks_position('none')
    ax_heatmap_f.set_title('Female lipid trajectories', fontsize=15)
    ax_heatmap_f.set_xlabel('Age (years)', fontsize=15)
    ax_heatmap_m.set_xticklabels([int(c) if c in np.arange(40, 75, 5) else '' for c in df.columns[df_f.shape[1]:]],
                                 rotation=0, ha='center', fontsize=15)
    ax_heatmap_m.xaxis.set_ticks_position('none')
    ax_heatmap_m.set_title('Male lipid trajectories', fontsize=15)
    ax_heatmap_m.set_xlabel('Age (years)', fontsize=15)

    colors = met_brewer.met_brew(name="Monet", n=labels_df.max() + 1, brew_type="continuous")
    sns.heatmap(pd.DataFrame(labels_df), cmap=colors, ax=ax_label, yticklabels=False, xticklabels=False, cbar=False)
    ax_label.set_ylabel('Lipid cluster', fontsize=15)
    ax_label.set_yticks([])

    def plot_cluster(c_id, ax, df_f, df_m, labels_df):
        for gender, df in zip(['Female', 'Male'], [df_f, df_m]):
            bottom = df.loc[labels_df[labels_df == c_id].index].quantile(0.1)
            upper = df.loc[labels_df[labels_df == c_id].index].quantile(0.9)
            middle = df.loc[labels_df[labels_df == c_id].index].quantile(0.5)
            ax.plot(df.columns, middle, label=gender)
            ax.fill_between(df.columns, bottom, upper, alpha=0.5)
            ax.set_ylim(-1, 1)
            ax.set_title('Cluster %d' % (c_id + 1))
            ax.set_yticks([-1, -.5, 0, .5, 1])
            add_text_at_corner(ax, 'n=%d' % (labels_df == c_id).sum(), 'top left')

        ax.legend(loc='lower left')

    for c_id, x1, x2, y1, y2 in zip(range(6),
                                    [7, 7, 7, 12, 12, 12],
                                    [11, 11, 11, 16, 16, 16],
                                    [1, 8, 15, 1, 8, 15],
                                    [6, 13, 20, 6, 13, 20]):
        ax = fig.add_subplot(gs[x1:x2, y1:y2])
        plot_cluster(c_id, ax, df_clustered.iloc[:, :df_f.shape[1]], df_clustered.iloc[:, df_f.shape[1]:], labels_df)

    fig.tight_layout()
    plt.savefig(os.path.join(loess_dir, 'figure.png'), dpi=200)

def compute_and_pred_lipid_cluster_median():
    loess_dir = mkdirifnotexists(os.path.join(work_dir, 'sex_specific_corrs', 'loess'))
    # labels_df = pd.read_csv(os.path.join(loess_dir, 'labels_df.csv'), index_col=0).rename(columns={'0': 'cluster'})
    labels_df = pd.read_csv(os.path.join(loess_dir, 'blood_lipids_labels_df_norm_per_sex.csv'), index_col=0).rename(columns={'0': 'cluster'})
    labels_df = labels_df.loc[labels_df['cluster'].isin(labels_df['cluster'].value_counts()[labels_df['cluster'].value_counts() > 1].index)]
    for sex in ['male', 'female']:
        lipids = pd.read_csv(os.path.join(Xs_dir, 'blood_lipids_%s_equal.csv' % sex), index_col=0)
        lipids = lipids.T.join(labels_df).dropna(subset=['cluster'])
        lipids_cluster_median = lipids.groupby('cluster').median().T#.drop([6.0], axis=1)
        lipids_cluster_median.rename(columns={c: 'cluster_%d'%c for c in lipids_cluster_median.columns}, inplace=True)
        lipids_cluster_median.index.names = ['RegistrationCode']
        lipids_cluster_median.to_csv(os.path.join(loess_dir, 'lipids_%s_cluster_median.csv' % sex))
    # return
    config = q_setup()

    save_dir = mkdirifnotexists(os.path.join(work_dir, 'sex_classification'))

    with config.qp(jobname='lipid_cluster_pred', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=5, max_u=200,
                   _mem_def='25G') as q:
        q.startpermanentrun()
        waiton = []

        res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
        x_names = res.loc[~res.index.str.contains('full')].index

        for x_name in x_names:
            for sex in ['male', 'female']:
                save_dir = mkdirifnotexists(os.path.join(loess_dir, 'lipid_cluster_median_prediction', x_name))
                output_path = os.path.join(save_dir, '%s-lipid_cluster_median/' % sex)
                x_path = os.path.join(Xs_dir, '%s_%s_equal.csv' % (x_name, sex))
                y_path = os.path.join(loess_dir, 'lipids_%s_cluster_median.csv' % sex)
                waiton.append(q.method(run_RSCV, (output_path, x_path, y_path, )))

        res = q.waitforresults(waiton)


# def os_sys_RandomizedSearchCV_old(output_path, path_to_X, path_to_Y, k_folds=5, subsample_X=1, random_seed=0,
#                               call_via_q=True):
#     os.system(' '.join(['python', '~/PycharmProjects/LabData/LabData/DataPredictors/RandomizedSearchCV.py',
#                         output_path,
#                         '-path_to_X', path_to_X,
#                         '-path_to_Y', path_to_Y,
#                         '-k_folds', str(k_folds),
#                         '-subsample_X', str(subsample_X),
#                         '-random_seed', str(random_seed),
#                         '' if call_via_q else '&']))
#     return

def os_sys_RandomizedSearchCV(output_path, path_to_X, path_to_Y, rscv_dic=rscv_kwargs, call_via_q=True):

    conf_path = create_pred_conf_file(output_path, path_to_X, path_to_Y, **rscv_dic)

    os.system(' '.join(['python', '~/PycharmProjects/LabData/LabData/DataPredictors/RandomizedSearchCV.py',
                        '--conf', conf_path,
                        '' if call_via_q else '&']))

    # os.system(' '.join(['python', '~/PycharmProjects/LabData/LabData/DataPredictors/RandomizedSearchCV.py',
    #                     output_path,
    #                     '-path_to_X', path_to_X,
    #                     '-path_to_Y', path_to_Y,
    #                     '-k_folds', str(k_folds),
    #                     '-subsample_X', str(subsample_X),
    #                     '-random_seed', str(random_seed),
    #                     '' if call_via_q else '&']))
    return





def predict_metabs(loaders=None):
    # predict metabs from diet and mb (and from genetics if possible)
    # use diet + age + sex
    # merge age with diet and mb and run models

    config = q_setup()

    serum_pred_dir = mkdirifnotexists(os.path.join(work_dir, 'blood_lipids_predictions'))

    metabs_age = pd.read_csv(os.path.join(Ys_dir, 'blood_lipids.csv'), index_col=0)
    path_to_Y = os.path.join(Xs_dir, 'blood_lipids.csv')
    if loaders is None:
        loaders = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0).index

    rcsv_dic = rscv_kwargs
    rcsv_dic['bootstrap'] = 'False'
    rcsv_dic['n_cols_per_job'] = 10

    with config.qp(jobname='lipid_pred', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, max_u=200,
                   _mem_def='25G') as q:
        q.startpermanentrun()
        waiton = []
        for data_name in loaders:
            # if data_name in ['diet', 'microbiome', 'serum_met']:
            output_path = os.path.join(serum_pred_dir, '%s-%s' % (data_name, 'blood_lipids'))

            if data_name in ['blood_lipids'] or os.path.exists(os.path.join(output_path, 'cv_results.csv')):
                continue
            print(data_name)
            if data_name == 'age_sex':
                data = pd.read_csv(os.path.join(Xs_dir, '%s.csv' % "blood_lipids"), index_col=0)[['gender']]
            else:
                data = pd.read_csv(os.path.join(Xs_dir, '%s.csv' % data_name), index_col=0)
            data_age_sex = data.join(metabs_age)
            path_to_X = os.path.join(serum_pred_dir, '%s_age_sex.csv' % data_name)
            data_age_sex.to_csv(path_to_X)



            # os_sys_RandomizedSearchCV(output_path, path_to_X, path_to_Y, 5, 1, 0, call_via_q=False)
            waiton.append(q.method(os_sys_RandomizedSearchCV, (output_path, path_to_X, path_to_Y, rcsv_dic, )))

        res = q.waitforresults(waiton)



def gender_classification_by_age(n_bootstraps=1000):
    config = q_setup()

    save_dir = mkdirifnotexists(os.path.join(work_dir, 'sex_classification'))

    with config.qp(jobname='sex_class', _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=5, max_u=200,
                   _mem_def='25G') as q:
        q.startpermanentrun()
        waiton = []

        res = pd.read_csv(os.path.join(pred_dir, 'results.csv'), index_col=0)
        x_names = res.loc[~res.index.str.contains('full')].index

        for x_name in x_names:
            # waiton.append(q.method(single_gender_classification_by_age, (x_name, n_bootstraps, )))
            waiton.append(q.method(run_RSCV_sex_classification, (x_name,)))

        res = q.waitforresults(waiton)


def single_gender_classification_by_age(x_name, n_bootstraps=1000):
    save_dir = mkdirifnotexists(os.path.join(work_dir, 'sex_classification', x_name))
    df_m = pd.read_csv(os.path.join(Xs_dir, '%s_female_equal.csv' % x_name), index_col=0)
    df_m['gender'] = 1
    df_f = pd.read_csv(os.path.join(Xs_dir, '%s_male_equal.csv' % x_name), index_col=0)
    df_f['gender'] = 0
    df_gender = pd.concat((df_m, df_f), axis=0)
    df_age = pd.read_csv(os.path.join(Ys_dir, '%s.csv' % x_name), index_col=0).loc[df_gender.index]

    y_gender = df_gender['gender']
    df_gender.drop('gender', axis=1, inplace=True)
    df_gender.dropna(how='all', axis=0, inplace=True)
    df_gender_isnull = df_gender.isnull().join(df_age)
    # df_gender = df_gender.fillna(df_gender.median()).apply(lambda x: (x - x.mean()) / x.std())
    df_age = df_gender.join(df_age)
    y_gender = y_gender.loc[df_age.index]

    model = LogisticRegressionCV(penalty='l1', cv=3, n_jobs=5, max_iter=500, solver='saga')
    model_null = LogisticRegressionCV(penalty='l1', cv=3, n_jobs=5, max_iter=500, solver='saga')

    X_train, X_test, y_train, y_test = train_test_split(df_age, y_gender, test_size=0.5, random_state=0)
    X_train_isnull, X_test_isnull, y_train, y_test = train_test_split(df_gender_isnull, y_gender, test_size=0.5,
                                                                      random_state=0)
    x_train_medians = X_train.median()
    X_train.fillna(x_train_medians, inplace=True)
    X_test.fillna(x_train_medians, inplace=True)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_train['age'] = X_train_isnull['age']
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    X_test['age'] = X_test_isnull['age']



    use_cols = df_gender_isnull.columns[
        ~(df_gender_isnull.apply(lambda x: spearmanr(x, y_gender, nan_policy='omit')[1]) < 0.01)]
    print(use_cols.shape)

    sample_weight = pd.concat((X_train['age'],
                               pd.cut(X_train['age'], np.arange(40, 80, 1)).rename('bin')), axis=1) \
        .merge(
        pd.DataFrame(1 / pd.cut(X_train['age'], np.arange(40, 80, 1)).value_counts().rename('freq')).reset_index(),
        left_on='bin', right_on='index')['freq']

    model.fit(X_train.loc[:, use_cols], y_train, sample_weight=sample_weight)
    model_null.fit(X_train_isnull.loc[:, use_cols], y_train, sample_weight=sample_weight)

    print(model.score(X_train.loc[:, use_cols], y_train))
    print(model.score(X_test.loc[:, use_cols], y_test))

    res_df = pd.DataFrame(index=['%s' % x_name, '%s (null)' % x_name], columns=['train', 'test'])
    res_df.loc['%s' % x_name, 'train'] = model.score(X_train.loc[:, use_cols], y_train)
    res_df.loc['%s' % x_name, 'test'] = model.score(X_test.loc[:, use_cols], y_test)
    res_df.loc['%s (null)' % x_name, 'train'] = model_null.score(X_train_isnull.loc[:, use_cols], y_train)
    res_df.loc['%s (null)' % x_name, 'test'] = model_null.score(X_test_isnull.loc[:, use_cols], y_test)
    res_df.to_csv(os.path.join(save_dir, 'results.csv'))


    def _bootstrapping(x_test, y_test, model):
        aucs = []
        y_score = model.predict(x_test)
        for i in range(n_bootstraps):
            bs1 = np.random.choice(range(x_test.shape[0]), size=x_test.shape[0])
            aucs.append(roc_auc_score(y_test.iloc[bs1], y_score[bs1]))
        return pd.Series(aucs).quantile([0.025, 0.975])

    age_bin = pd.cut(X_test['age'], [40, 45, 50, 55, 60, 65, 75])
    res_df = pd.DataFrame(index=age_bin.dropna().unique().sort_values(), columns=['auc', '2.5', '97.5', 'size'])
    for ab in age_bin.dropna().unique().sort_values():
        res_df.loc[ab, 'size'] = y_test.loc[age_bin == ab].shape[0]
        y_score = model.predict(X_test.loc[age_bin == ab].loc[:, use_cols])
        a, b = _bootstrapping(X_test.loc[age_bin == ab].loc[:, use_cols], y_test.loc[age_bin == ab], model)
        res_df.loc[ab, 'auc'] = roc_auc_score(y_test.loc[age_bin == ab], y_score)
        res_df.loc[ab, '2.5'] = a
        res_df.loc[ab, '97.5'] = b
    res_df.to_csv(os.path.join(save_dir, 'results_by_age_full_range.csv'))

    age_bin = pd.cut(X_test['age'], [40, 50, 75])
    res_df = pd.DataFrame(index=age_bin.dropna().unique().sort_values(), columns=['auc', '2.5', '97.5', 'size'])
    for ab in age_bin.dropna().unique().sort_values():
        res_df.loc[ab, 'size'] = y_test.loc[age_bin == ab].shape[0]
        y_score = model.predict(X_test.loc[age_bin == ab].loc[:, use_cols])
        a, b = _bootstrapping(X_test.loc[age_bin == ab].loc[:, use_cols], y_test.loc[age_bin == ab], model)
        res_df.loc[ab, 'auc'] = roc_auc_score(y_test.loc[age_bin == ab], y_score)
        res_df.loc[ab, '2.5'] = a
        res_df.loc[ab, '97.5'] = b
    res_df.to_csv(os.path.join(save_dir, 'results_by_age_menopause.csv'))

def run_RSCV_sex_classification(x_name):
    save_dir = mkdirifnotexists(os.path.join(work_dir, 'sex_classification', x_name))
    # read males and females equal
    df_m = pd.read_csv(os.path.join(Xs_dir, '%s_female_equal.csv' % x_name), index_col=0)
    df_m['gender'] = 1
    df_f = pd.read_csv(os.path.join(Xs_dir, '%s_male_equal.csv' % x_name), index_col=0)
    df_f['gender'] = 0
    df_gender = pd.concat((df_m, df_f), axis=0)
    df_age = pd.read_csv(os.path.join(Ys_dir, '%s.csv' % x_name), index_col=0).loc[df_gender.index]

    # put gender aside as y
    y_gender = df_gender['gender']
    df_gender.drop('gender', axis=1, inplace=True)
    df_gender.dropna(how='all', axis=0, inplace=True)
    df_gender_isnull = df_gender.isnull().join(df_age)
    df_age = df_gender.join(df_age)
    y_gender = y_gender.loc[df_age.index]

    # remove features that their missing patterns correlate with gender
    use_cols = df_gender_isnull.columns[
        ~(df_gender_isnull.apply(lambda x: spearmanr(x, y_gender, nan_policy='omit')[1]) < 0.01)]
    print(use_cols.shape)

    # save to disk both x and y to a new directory
    df_age.loc[:, use_cols].to_csv(os.path.join(save_dir, '%s_X.csv' % x_name))
    df_gender_isnull.loc[:, use_cols].to_csv(os.path.join(save_dir, '%s_X_isnull.csv' % x_name))
    y_gender.to_csv(os.path.join(save_dir, 'y.csv'))

    # for x in ['X', 'X_isnull']:  # TODO: if one collapse, the otherone won't run
    for x in ['X']: #  X X_isnull
        output_path = os.path.join(save_dir, '%s-sex/' % x)
        x_path = os.path.join(save_dir, '%s_%s.csv' % (x_name, x))
        y_path = os.path.join(save_dir, 'y.csv')

        run_RSCV(output_path, x_path, y_path)
    return

def sex_class_results_by_age_group(n_bootstraps=1000, missing=False):
    if missing:
        missing = '_isnull'
    else:
        missing = ''
    x_names = [f for f in os.listdir(os.path.join(work_dir, 'sex_classification'))]
    for x_name in x_names:
        print(x_name)
        save_dir = mkdirifnotexists(os.path.join(work_dir, 'sex_classification', x_name))
        y_true = pd.read_csv(os.path.join(save_dir, 'y.csv'), index_col=0)['gender']
        y_pred = pd.read_csv(os.path.join(save_dir, 'X%s-sex' % missing, 'cv_predictions_df.csv'), index_col=0)['gender'].loc[y_true.index]
        age = pd.read_csv(os.path.join(Ys_dir, '%s.csv' % x_name), index_col=0).reindex(y_true.index)
        def _bootstrapping(y_true, y_pred):
            aucs = []
            for i in range(n_bootstraps):
                bs1 = np.random.choice(range(y_true.shape[0]), size=y_true.shape[0])
                aucs.append(roc_auc_score(y_true.iloc[bs1], y_pred[bs1]))
            return pd.Series(aucs).quantile([0.025, 0.975])

        age_bin = pd.cut(age['age'], [40, 45, 50, 55, 60, 65, 75])
        res_df = pd.DataFrame(index=age_bin.dropna().unique().sort_values(), columns=['auc', '2.5', '97.5', 'size'])
        for ab in age_bin.dropna().unique().sort_values():
            res_df.loc[ab, 'size'] = y_true.loc[age_bin == ab].shape[0]
            a, b = _bootstrapping(y_true.loc[age_bin == ab], y_pred.loc[age_bin == ab])
            res_df.loc[ab, 'auc'] = roc_auc_score(y_true.loc[age_bin == ab], y_pred.loc[age_bin == ab])
            res_df.loc[ab, '2.5'] = a
            res_df.loc[ab, '97.5'] = b

        res_df.to_csv(os.path.join(save_dir, 'LGBM_results_by_age_full_range%s.csv' % missing))



def main():

    # build_Xs_and_Ys()

    # loaders = ['blood_lipids']
    # loaders = ['liver', 'cardiovascular_system', 'renal_function', 'lifestyle', 'hematopoietic_system', 'diet_questions']
    # loaders = None
    # subsample_X = [0.5, 0.5]
    # subsample_X = None
    # run prediction models for separated sexes
    # run_age_predictions(only_per_sex=True, loaders=loaders)
    # run prediction models for combined sexes
    # run_age_predictions(only_per_sex=False, loaders=loaders)

    # bins = [40, 45, 50, 55, 60, 65, 70]
    # concat_results_and_compute_residuals(bins=bins)
    # # # # # #
    # plot_r2_summary_barplot()
    # #
    rank_or_z = 'centered'  # 'rank' 'zscore'

    # save_residuals(rank_or_z=rank_or_z)
    # plot_true_vs_permuted_mean_residuals(rank_or_z=rank_or_z)
    # #
    # plot_residuals_correlation_matrix(rank_or_z=rank_or_z)
    # #
    # plot_residuals_edges_clustermap(rank_or_z=rank_or_z)

    # tsne_and_counts_over_edges(rank_or_z=rank_or_z)

    correlate_residuals_with_medical_conditions(rank_or_z=rank_or_z)
    # #
    # plot_running_scores()
    # #
    # compare_shap_values()
    # compare_shap_values_subsample('subsample_X')
    # distance_metric = 'spearman'  # 'spearman', 'euclidean'
    # compute_all_mantel_tests(distance_metric=distance_metric)

    # bootstrap_sex_spec_corrs()

    # analyze_sex_spec_corrs()
    # x_name = 'microbiome'
    # loess_age_vs_metabs(x_name = x_name)

    # gender_stratified_lipid_trajectories()

    # predict_metabs(['age_sex'])
    # predict_metabs()

    # gender_classification_by_age()
    # sex_class_results_by_age_group(missing=True)

    # compute_and_pred_lipid_cluster_median()

    return


if __name__ == "__main__":
    main()

