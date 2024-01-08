import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor, RandomForestRegressor, \
    HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor, SGDRegressor, GammaRegressor, \
    PoissonRegressor, PassiveAggressiveRegressor, TheilSenRegressor, BayesianRidge, ARDRegression, TweedieRegressor, \
    PassiveAggressiveClassifier, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm
from sklearn.model_selection import cross_validate
import xgboost
import lightgbm
import catboost

all = pd.read_csv('../data/raw/all.csv')
all = all.drop(columns=['IMR', 'ID'])

all['MRG'] = all['MRG'].apply(lambda x: x if x in [91, 92, 93, 94, 95, 96, 97] else 90)

for k in all['REL'].value_counts().keys()[-1:]:
    all = all[all['REL'] != k]

for k in all['WORKPLACE'].value_counts().keys()[-2:]:
    all = all[all['WORKPLACE'] != k]

ITM40_mean, ITM40_std = all['ITM40'].mean(), all['ITM40'].std()
all['ITM40'] = (all['ITM40'] - ITM40_mean) / ITM40_std

X, Y = all.iloc[:, :-1], all.iloc[:, -1]
df_encoded = pd.get_dummies(X, columns=list(X.columns), prefix='Prefix')

model_ExtraTreesRegressor = ExtraTreesRegressor()
model_RandomForestRegressor = RandomForestRegressor()
model_BaggingRegressor = BaggingRegressor()
model_HistGradientBoostingRegressor = HistGradientBoostingRegressor()
model_GradientBoostingRegressor = GradientBoostingRegressor()
model_DecisionTreeRegressor = DecisionTreeRegressor()
model_SVR = SVR(max_iter=1000)
model_NuSVR = NuSVR(max_iter=1000)
model_ExtraTreeRegressor = ExtraTreeRegressor()
model_MLPRegressor = MLPRegressor(max_iter=1000)
model_KNeighborsRegressor = KNeighborsRegressor()
model_LinearSVR = LinearSVR(max_iter=1000)
model_HuberRegressor = HuberRegressor()
model_TheilSenRegressor = TheilSenRegressor()
model_BayesianRidge = BayesianRidge()
model_KernelRidge = KernelRidge()
model_LinearRegression = LinearRegression()
model_SGDRegressor = SGDRegressor()
model_ARDRegression = ARDRegression()
model_PoissonRegressor = PoissonRegressor()
model_GammaRegressor = GammaRegressor()
model_RANSACRegressor = RANSACRegressor()
model_PLSRegression = PLSRegression()
model_PassiveAggressiveRegressor = PassiveAggressiveRegressor()
model_TweedieRegressor = TweedieRegressor()
model_AdaBoostRegressor = AdaBoostRegressor()
model_PassiveAggressiveClassifier = PassiveAggressiveClassifier()
model_ElasticNet = ElasticNet()
model_GaussianProcessRegressor = GaussianProcessRegressor()
model_XGBRegressor = xgboost.XGBRegressor()
model_LGBMRegressor = lightgbm.LGBMRegressor()
model_CatBoostRegressor = catboost.CatBoostRegressor()

model_list = [
    model_ExtraTreesRegressor,
    model_RandomForestRegressor,
    model_BaggingRegressor,
    model_HistGradientBoostingRegressor,
    model_GradientBoostingRegressor,
    model_DecisionTreeRegressor,
    model_SVR,
    model_NuSVR,
    model_ExtraTreeRegressor,
    model_MLPRegressor,
    model_KNeighborsRegressor,
    model_LinearSVR,
    model_HuberRegressor,
    model_TheilSenRegressor,
    model_BayesianRidge,
    model_KernelRidge,
    model_LinearRegression,
    model_SGDRegressor,
    model_ARDRegression,
    model_PoissonRegressor,
    model_GammaRegressor,
    model_RANSACRegressor,
    model_PLSRegression,
    model_PassiveAggressiveRegressor,
    model_TweedieRegressor,
    model_AdaBoostRegressor,
    model_PassiveAggressiveClassifier,
    model_ElasticNet,
    model_GaussianProcessRegressor,
    model_XGBRegressor,
    model_LGBMRegressor,
    model_CatBoostRegressor,
]

print('running...')

result = []
for model in tqdm(model_list):
    try:
        res = cross_validate(model, df_encoded, Y, n_jobs=-1, verbose=0, return_train_score=True,
                             scoring='neg_mean_absolute_error')
        test_score = -res['test_score'].mean()
        train_score = -res['train_score'].mean()
        result.append([model.__class__.__name__, test_score, train_score])
    except Exception as e:
        print(e)

pd.DataFrame(result, columns=["name", "test_score", "train_score"]).to_csv("mae_models.csv", index=False)
