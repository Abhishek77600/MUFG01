import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class AdvancedAlgorithmSuite:
    """
    Comprehensive algorithm suite for heart disease prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.ensemble_models = {}
        
    def get_base_algorithms(self):
        """
        Get base algorithms with optimized default parameters
        """
        algorithms = {
            # Tree-based models
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'extra_trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            
            # Modern gradient boosting
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            ),
            
            'catboost': CatBoostClassifier(
                iterations=200,
                learning_rate=0.1,
                depth=6,
                random_state=self.random_state,
                verbose=False
            ),
            
            # Linear models
            'logistic_regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=self.random_state,
                max_iter=1000
            ),
            
            'ridge_classifier': RidgeClassifier(
                alpha=1.0,
                random_state=self.random_state
            ),
            
            # Support Vector Machines
            'svm_rbf': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            
            'svm_linear': SVC(
                C=1.0,
                kernel='linear',
                probability=True,
                random_state=self.random_state
            ),
            
            # Neural Networks
            'mlp_classifier': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            ),
            
            # Instance-based
            'knn': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                algorithm='auto'
            ),
            
            # Probabilistic
            'naive_bayes': GaussianNB(),
            
            # Discriminant Analysis
            'lda': LinearDiscriminantAnalysis(),
            'qda': QuadraticDiscriminantAnalysis(),
            
            # Bagging
            'bagging': BaggingClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=10),
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        return algorithms
    
    def get_hyperparameter_grids(self):
        """
        Get comprehensive hyperparameter grids for optimization
        """
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            
            'extra_trees': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            },
            
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            },
            
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_child_samples': [10, 20, 30],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            },
            
            'catboost': {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 5, 7, 10],
                'l2_leaf_reg': [1, 3, 5, 7],
                'border_count': [32, 64, 128]
            },
            
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'l1_ratio': [0.1, 0.5, 0.9]
            },
            
            'ridge_classifier': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            
            'svm_rbf': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf']
            },
            
            'svm_linear': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear']
            },
            
            'mlp_classifier': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100), (200, 100)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                'p': [1, 2]
            },
            
            'bagging': {
                'n_estimators': [50, 100, 200],
                'max_samples': [0.5, 0.7, 1.0],
                'max_features': [0.5, 0.7, 1.0]
            }
        }
        
        return param_grids
    
    def train_base_models(self, X, y):
        """
        Train all base models with default parameters
        """
        self.models = self.get_base_algorithms()
        
        print(f"Training {len(self.models)} base models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            try:
                model.fit(X, y)
                print(f"    ✓ {name} trained successfully")
            except Exception as e:
                print(f"    ✗ {name} failed: {str(e)}")
                
        return self.models
    
    def optimize_hyperparameters(self, X, y, cv=5, scoring='roc_auc', n_jobs=-1, use_randomized=True):
        """
        Optimize hyperparameters using GridSearch or RandomizedSearch
        """
        algorithms = self.get_base_algorithms()
        param_grids = self.get_hyperparameter_grids()
        
        self.best_models = {}
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scorer = make_scorer(roc_auc_score, needs_proba=True)
        
        print(f"Optimizing hyperparameters for {len(algorithms)} models...")
        
        for name, model in algorithms.items():
            if name in param_grids:
                print(f"  Optimizing {name}...")
                
                try:
                    if use_randomized:
                        search = RandomizedSearchCV(
                            estimator=model,
                            param_distributions=param_grids[name],
                            n_iter=50,  # Number of parameter settings sampled
                            scoring=scorer,
                            cv=cv_splitter,
                            n_jobs=n_jobs,
                            random_state=self.random_state,
                            verbose=0
                        )
                    else:
                        search = GridSearchCV(
                            estimator=model,
                            param_grid=param_grids[name],
                            scoring=scorer,
                            cv=cv_splitter,
                            n_jobs=n_jobs,
                            verbose=0
                        )
                    
                    search.fit(X, y)
                    
                    self.best_models[name] = {
                        'best_estimator': search.best_estimator_,
                        'best_params': search.best_params_,
                        'best_score': search.best_score_,
                        'cv_results': search.cv_results_
                    }
                    
                    print(f"    ✓ {name} - Best CV Score: {search.best_score_:.4f}")
                    
                except Exception as e:
                    print(f"    ✗ {name} optimization failed: {str(e)}")
                    # Fallback to default model
                    self.best_models[name] = {
                        'best_estimator': model.fit(X, y),
                        'best_params': {},
                        'best_score': 0.0,
                        'cv_results': {}
                    }
        
        return self.best_models
    
    def create_advanced_ensembles(self, X, y, top_n=5):
        """
        Create advanced ensemble models
        """
        if not self.best_models:
            raise ValueError("Must optimize hyperparameters first!")
        
        # Sort models by CV score
        sorted_models = sorted(
            self.best_models.items(), 
            key=lambda x: x[1]['best_score'], 
            reverse=True
        )
        
        top_models = sorted_models[:top_n]
        print(f"Creating ensembles from top {len(top_models)} models:")
        for name, info in top_models:
            print(f"  {name}: {info['best_score']:.4f}")
        
        self.ensemble_models = {}
        
        # 1. Voting Classifiers
        estimators = [(name, info['best_estimator']) for name, info in top_models]
        
        # Hard voting
        voting_hard = VotingClassifier(estimators=estimators, voting='hard')
        voting_hard.fit(X, y)
        self.ensemble_models['voting_hard'] = voting_hard
        
        # Soft voting
        voting_soft = VotingClassifier(estimators=estimators, voting='soft')
        voting_soft.fit(X, y)
        self.ensemble_models['voting_soft'] = voting_soft
        
        # 2. Stacking Classifiers
        # Two-level stacking
        base_estimators = [(name, info['best_estimator']) for name, info in top_models[:3]]
        
        # Stacking with Logistic Regression
        stacking_lr = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=self.random_state),
            cv=5,
            n_jobs=-1
        )
        stacking_lr.fit(X, y)
        self.ensemble_models['stacking_lr'] = stacking_lr
        
        # Stacking with Random Forest
        stacking_rf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            cv=5,
            n_jobs=-1
        )
        stacking_rf.fit(X, y)
        self.ensemble_models['stacking_rf'] = stacking_rf
        
        # 3. Weighted Ensemble (based on CV scores)
        weights = [info['best_score'] for name, info in top_models]
        weights = np.array(weights) / np.sum(weights)  # Normalize
        
        voting_weighted = VotingClassifier(
            estimators=estimators, 
            voting='soft',
            weights=weights
        )
        voting_weighted.fit(X, y)
        self.ensemble_models['voting_weighted'] = voting_weighted
        
        print(f"Created {len(self.ensemble_models)} ensemble models")
        return self.ensemble_models
    
    def get_model_summary(self):
        """
        Get summary of all trained models
        """
        summary = {
            'base_models': len(self.models),
            'optimized_models': len(self.best_models),
            'ensemble_models': len(self.ensemble_models)
        }
        
        if self.best_models:
            best_scores = {name: info['best_score'] for name, info in self.best_models.items()}
            summary['best_individual_model'] = max(best_scores, key=best_scores.get)
            summary['best_individual_score'] = max(best_scores.values())
        
        return summary

# Example usage
def run_advanced_algorithm_suite(X, y):
    """
    Complete algorithm suite execution
    """
    suite = AdvancedAlgorithmSuite(random_state=42)
    
    # 1. Train base models
    base_models = suite.train_base_models(X, y)
    
    # 2. Optimize hyperparameters
    best_models = suite.optimize_hyperparameters(X, y, use_randomized=True)
    
    # 3. Create ensembles
    ensemble_models = suite.create_advanced_ensembles(X, y, top_n=5)
    
    # 4. Get summary
    summary = suite.get_model_summary()
    
    print("\n" + "="*60)
    print("ADVANCED ALGORITHM SUITE SUMMARY")
    print("="*60)
    print(f"Base models trained: {summary['base_models']}")
    print(f"Optimized models: {summary['optimized_models']}")
    print(f"Ensemble models: {summary['ensemble_models']}")
    
    if 'best_individual_model' in summary:
        print(f"Best individual model: {summary['best_individual_model']}")
        print(f"Best CV score: {summary['best_individual_score']:.4f}")
    
    return suite

if __name__ == "__main__":
    # Example usage
    print("Advanced Algorithm Suite for Heart Disease Prediction")
    print("This module provides comprehensive ML algorithms including:")
    print("- Traditional ML: RF, SVM, KNN, NB, LDA/QDA")
    print("- Modern Boosting: XGBoost, LightGBM, CatBoost")
    print("- Neural Networks: MLP")
    print("- Advanced Ensembles: Voting, Stacking, Weighted")
    print("- Hyperparameter optimization with RandomizedSearch")