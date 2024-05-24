from django import forms
from django.contrib.auth.models import User
from .models import Profile


class DLClassificationForm(forms.Form):
    """Class for creating DL classification form."""

    # --- DATA AUGMENTATION ---

    dpcla_data_augmentation_cropping = forms.BooleanField(required=False)
    dpcla_data_augmentation_horizontal_flip = forms.BooleanField(required=False)
    dpcla_data_augmentation_vertical_flip = forms.BooleanField(required=False)
    dpcla_data_augmentation_translation = forms.BooleanField(required=False)
    dpcla_data_augmentation_rotation = forms.BooleanField(required=False)
    dpcla_data_augmentation_zoom = forms.BooleanField(required=False)
    dpcla_data_augmentation_contrast = forms.BooleanField(required=False)
    dpcla_data_augmentation_brightness = forms.BooleanField(required=False)

    # --- MODELS ---

    # Xception #
    dpcla_xception = forms.BooleanField(required=False)

    # VGG
    dpcla_vgg_11 = forms.BooleanField(required=False)
    dpcla_vgg_13 = forms.BooleanField(required=False)
    dpcla_vgg_16 = forms.BooleanField(required=False)
    dpcla_vgg_19 = forms.BooleanField(required=False)

    # ResNet, ResNet V2, ResNetRS
    dpcla_resnet_18 = forms.BooleanField(required=False)
    dpcla_resnet_34 = forms.BooleanField(required=False)
    dpcla_resnet_50 = forms.BooleanField(required=False)
    dpcla_resnet_50_v2 = forms.BooleanField(required=False)
    dpcla_resnet_rs_50 = forms.BooleanField(required=False)
    dpcla_resnet_101 = forms.BooleanField(required=False)
    dpcla_resnet_101_v2 = forms.BooleanField(required=False)
    dpcla_resnet_rs_101 = forms.BooleanField(required=False)
    dpcla_resnet_152 = forms.BooleanField(required=False)
    dpcla_resnet_152_v2 = forms.BooleanField(required=False)
    dpcla_resnet_rs_152 = forms.BooleanField(required=False)
    dpcla_resnet_rs_200 = forms.BooleanField(required=False)
    dpcla_resnet_rs_270 = forms.BooleanField(required=False)
    dpcla_resnet_rs_350 = forms.BooleanField(required=False)
    dpcla_resnet_rs_420 = forms.BooleanField(required=False)

    # Inception
    dpcla_inception_v3 = forms.BooleanField(required=False)
    dpcla_inception_resnet_v2 = forms.BooleanField(required=False)

    # MobileNet
    dpcla_mobilenet = forms.BooleanField(required=False)
    dpcla_mobilenet_v2 = forms.BooleanField(required=False)
    dpcla_mobilenet_v3_small = forms.BooleanField(required=False)
    dpcla_mobilenet_v3_large = forms.BooleanField(required=False)

    # DenseNet
    dpcla_densenet_121 = forms.BooleanField(required=False)
    dpcla_densenet_169 = forms.BooleanField(required=False)
    dpcla_densenet_201 = forms.BooleanField(required=False)

    # NasNet
    dpcla_nasnet_mobile = forms.BooleanField(required=False)
    dpcla_nasnet_large = forms.BooleanField(required=False)

    # EfficientNet, EfficientNet V2
    dpcla_efficientnet_b0 = forms.BooleanField(required=False)
    dpcla_efficientnet_b0_v2 = forms.BooleanField(required=False)
    dpcla_efficientnet_b1 = forms.BooleanField(required=False)
    dpcla_efficientnet_b1_v2 = forms.BooleanField(required=False)
    dpcla_efficientnet_b2 = forms.BooleanField(required=False)
    dpcla_efficientnet_b2_v2 = forms.BooleanField(required=False)
    dpcla_efficientnet_b3 = forms.BooleanField(required=False)
    dpcla_efficientnet_b3_v2 = forms.BooleanField(required=False)
    dpcla_efficientnet_b4 = forms.BooleanField(required=False)
    dpcla_efficientnet_b5 = forms.BooleanField(required=False)
    dpcla_efficientnet_b6 = forms.BooleanField(required=False)
    dpcla_efficientnet_b7 = forms.BooleanField(required=False)
    dpcla_efficientnet_v2_small = forms.BooleanField(required=False)
    dpcla_efficientnet_v2_medium = forms.BooleanField(required=False)
    dpcla_efficientnet_v2_large = forms.BooleanField(required=False)

    # ConvNeXT
    dpcla_convnext_tiny = forms.BooleanField(required=False)
    dpcla_convnext_small = forms.BooleanField(required=False)
    dpcla_convnext_base = forms.BooleanField(required=False)
    dpcla_convnext_large = forms.BooleanField(required=False)
    dpcla_convnext_xlarge = forms.BooleanField(required=False)

    # RegNetX, RegNetY
    dpcla_regnet_x_002 = forms.BooleanField(required=False)
    dpcla_regnet_y_002 = forms.BooleanField(required=False)
    dpcla_regnet_x_004 = forms.BooleanField(required=False)
    dpcla_regnet_y_004 = forms.BooleanField(required=False)
    dpcla_regnet_x_006 = forms.BooleanField(required=False)
    dpcla_regnet_y_006 = forms.BooleanField(required=False)
    dpcla_regnet_x_008 = forms.BooleanField(required=False)
    dpcla_regnet_y_008 = forms.BooleanField(required=False)
    dpcla_regnet_x_016 = forms.BooleanField(required=False)
    dpcla_regnet_y_016 = forms.BooleanField(required=False)
    dpcla_regnet_x_032 = forms.BooleanField(required=False)
    dpcla_regnet_y_032 = forms.BooleanField(required=False)
    dpcla_regnet_x_040 = forms.BooleanField(required=False)
    dpcla_regnet_y_040 = forms.BooleanField(required=False)
    dpcla_regnet_x_064 = forms.BooleanField(required=False)
    dpcla_regnet_y_064 = forms.BooleanField(required=False)
    dpcla_regnet_x_080 = forms.BooleanField(required=False)
    dpcla_regbet_y_080 = forms.BooleanField(required=False)
    dpcla_regnet_x_120 = forms.BooleanField(required=False)
    dpcla_regnet_y_120 = forms.BooleanField(required=False)
    dpcla_regnet_x_160 = forms.BooleanField(required=False)
    dpcla_regnet_y_160 = forms.BooleanField(required=False)
    dpcla_regnet_x_320 = forms.BooleanField(required=False)
    dpcla_regnet_y_320 = forms.BooleanField(required=False)

    # --- TRAINING STRATEGY ---

    dpcla_optimizer1 = forms.CharField(max_length=100, required=True)
    dpcla_optimizer2 = forms.CharField(max_length=100, required=False)
    dpcla_optimizer3 = forms.CharField(max_length=100, required=False)

    dpcla_loss1 = forms.CharField(max_length=100, required=True)
    dpcla_loss2 = forms.CharField(max_length=100, required=False)
    dpcla_loss3 = forms.CharField(max_length=100, required=False)

    dpcla_lr1 = forms.FloatField(required=True)
    dpcla_lr2 = forms.FloatField(required=False)
    dpcla_lr3 = forms.FloatField(required=False)

    # --- EXPLAINABILITY (XAI) ---

    dpcla_activationmaximization = forms.BooleanField(required=False)
    dpcla_gradcam = forms.BooleanField(required=False)
    dpcla_gradcamplusplus = forms.BooleanField(required=False)
    dpcla_scorecam = forms.BooleanField(required=False)
    dpcla_fasterscorecam = forms.BooleanField(required=False)
    dpcla_layercam = forms.BooleanField(required=False)
    dpcla_vanillasaliency = forms.BooleanField(required=False)
    dpcla_smoothgrad = forms.BooleanField(required=False)

    # --- OUTPUT ---

    dpcla_savemodel = forms.BooleanField(required=False)
    dpcla_traingraph = forms.BooleanField(required=False)
    dpcla_confmatrix = forms.BooleanField(required=False)
    dpcla_classreport = forms.BooleanField(required=False)
    dpcla_tflite = forms.BooleanField(required=False)


class DLSegmentation(forms.Form):
    """Class for creating DL segmentation form."""

    dpseg_unet = forms.BooleanField(required=False)
    dpseg_unetplusplus = forms.BooleanField(required=False)
    dpseg_manet = forms.BooleanField(required=False)
    dpseg_linknet = forms.BooleanField(required=False)
    dpseg_fpn = forms.BooleanField(required=False)
    dpseg_pspnet = forms.BooleanField(required=False)
    dpseg_pan = forms.BooleanField(required=False)
    dpseg_deeplabv3 = forms.BooleanField(required=False)
    dpseg_deeplabv3plus = forms.BooleanField(required=False)


class MLClassificationForm(forms.Form):
    """Class for creating ML classification form."""

    mlcla_lr = forms.BooleanField(required=False)
    mlcla_knn = forms.BooleanField(required=False)
    mlcla_nb = forms.BooleanField(required=False)
    mlcla_dt = forms.BooleanField(required=False)
    mlcla_svm = forms.BooleanField(required=False)
    mlcla_rbfsvm = forms.BooleanField(required=False)
    mlcla_gpc = forms.BooleanField(required=False)
    mlcla_mlp = forms.BooleanField(required=False)
    mlcla_ridge = forms.BooleanField(required=False)
    mlcla_rf = forms.BooleanField(required=False)
    mlcla_qda = forms.BooleanField(required=False)
    mlcla_ada = forms.BooleanField(required=False)
    mlcla_gbc = forms.BooleanField(required=False)
    mlcla_lda = forms.BooleanField(required=False)
    mlcla_et = forms.BooleanField(required=False)
    mlcla_xgboost = forms.BooleanField(required=False)
    mlcla_lightgbm = forms.BooleanField(required=False)
    mlcla_catboost = forms.BooleanField(required=False)
    mlcla_dummy = forms.BooleanField(required=False)


class MLRegressionForm(forms.Form):
    """Class for creating mL regression form."""

    mlreg_lr = forms.BooleanField(required=False)
    mlreg_lasso = forms.BooleanField(required=False)
    mlreg_ridge = forms.BooleanField(required=False)
    mlreg_en = forms.BooleanField(required=False)
    mlreg_lar = forms.BooleanField(required=False)
    mlreg_llar = forms.BooleanField(required=False)
    mlreg_omp = forms.BooleanField(required=False)
    mlreg_br = forms.BooleanField(required=False)
    mlreg_ard = forms.BooleanField(required=False)
    mlreg_par = forms.BooleanField(required=False)
    mlreg_ransac = forms.BooleanField(required=False)
    mlreg_tr = forms.BooleanField(required=False)
    mlreg_huber = forms.BooleanField(required=False)
    mlreg_kr = forms.BooleanField(required=False)
    mlreg_svm = forms.BooleanField(required=False)
    mlreg_knn = forms.BooleanField(required=False)
    mlreg_dt = forms.BooleanField(required=False)
    mlreg_rf = forms.BooleanField(required=False)
    mlreg_et = forms.BooleanField(required=False)
    mlreg_ada = forms.BooleanField(required=False)
    mlreg_gbr = forms.BooleanField(required=False)
    mlreg_mlp = forms.BooleanField(required=False)
    mlreg_xgboost = forms.BooleanField(required=False)
    mlreg_lightgbm = forms.BooleanField(required=False)
    mlreg_catboost = forms.BooleanField(required=False)
    mlreg_dummy = forms.BooleanField(required=False)
    mlreg_bagging = forms.BooleanField(required=False)
    mlreg_stacking = forms.BooleanField(required=False)
    mlreg_voting = forms.BooleanField(required=False)


class MLTimeSeriesForm(forms.Form):
    """Class for creating ML time series form."""

    mlts_naive = forms.BooleanField(required=False)
    mlts_grand_means = forms.BooleanField(required=False)
    mlts_snaive = forms.BooleanField(required=False)
    mlts_polytrend = forms.BooleanField(required=False)
    mlts_arima = forms.BooleanField(required=False)
    mlts_auto_arima = forms.BooleanField(required=False)
    mlts_exp_smooth = forms.BooleanField(required=False)
    mlts_ets = forms.BooleanField(required=False)
    mlts_theta = forms.BooleanField(required=False)
    mlts_stlf = forms.BooleanField(required=False)
    mlts_croston = forms.BooleanField(required=False)
    mlts_tbats = forms.BooleanField(required=False)
    mlts_bats = forms.BooleanField(required=False)
    mlts_prophet = forms.BooleanField(required=False)
    mlts_lr_cds_dt = forms.BooleanField(required=False)
    mlts_en_cds_dt = forms.BooleanField(required=False)
    mlts_ridge_cds_dt = forms.BooleanField(required=False)
    mlts_lasso_cds_dt = forms.BooleanField(required=False)
    mlts_lar_cds_dt = forms.BooleanField(required=False)
    mlts_llar_cds_dt = forms.BooleanField(required=False)
    mlts_br_cds_dt = forms.BooleanField(required=False)
    mlts_huber_cds_dt = forms.BooleanField(required=False)
    mlts_par_cds_dt = forms.BooleanField(required=False)
    mlts_omp_cds_dt = forms.BooleanField(required=False)
    mlts_knn_cds_dt = forms.BooleanField(required=False)
    mlts_dt_cds_dt = forms.BooleanField(required=False)
    mlts_rf_cds_dt = forms.BooleanField(required=False)
    mlts_et_cds_dt = forms.BooleanField(required=False)
    mlts_gbr_cds_dt = forms.BooleanField(required=False)
    mlts_ada_cds_dt = forms.BooleanField(required=False)
    mlts_xgboost_cds_dt = forms.BooleanField(required=False)
    mlts_lightgbm_cds_dt = forms.BooleanField(required=False)
    mlts_catboost_cds_dt = forms.BooleanField(required=False)


class MLClusteringForm(forms.Form):
    """Class for creating ML Clustering form."""

    mlclu_kmeans = forms.BooleanField(required=False)
    mlclu_ap = forms.BooleanField(required=False)
    mlclu_meanshift = forms.BooleanField(required=False)
    mlclu_sc = forms.BooleanField(required=False)
    mlclu_hclust = forms.BooleanField(required=False)
    mlclu_dbscan = forms.BooleanField(required=False)
    mlclu_optics = forms.BooleanField(required=False)
    mlclu_birch = forms.BooleanField(required=False)
    mlclu_kmodes = forms.BooleanField(required=False)


class MLAnomalyDetectionForm(forms.Form):
    """Class for creating ML Anomaly Detection form."""

    mlad_abod = forms.BooleanField(required=False)
    mlad_cluster = forms.BooleanField(required=False)
    mlad_cof = forms.BooleanField(required=False)
    mlad_iforest = forms.BooleanField(required=False)
    mlad_histogram = forms.BooleanField(required=False)
    mlad_knn = forms.BooleanField(required=False)
    mlad_lof = forms.BooleanField(required=False)
    mlad_svm = forms.BooleanField(required=False)
    mlad_pca = forms.BooleanField(required=False)
    mlad_mcd = forms.BooleanField(required=False)
    mlad_sod = forms.BooleanField(required=False)
    mlad_sos = forms.BooleanField(required=False)


class NLPTextGenerationForm(forms.Form):
    """Class for creating NLP Text Generation form."""

    # Alfred
    nlptg_model_alfred_40b = forms.BooleanField(required=False)

    # Code
    nlptg_model_code_13b = forms.BooleanField(required=False)
    nlptg_model_code_33b = forms.BooleanField(required=False)

    # CodeLLaMA Models
    nlptg_model_codellama_7b = forms.BooleanField(required=False)
    nlptg_model_codellama_7b_instruct = forms.BooleanField(required=False)
    nlptg_model_codellama_7b_python = forms.BooleanField(required=False)
    nlptg_model_codellama_13b = forms.BooleanField(required=False)
    nlptg_model_codellama_13b_instruct = forms.BooleanField(required=False)
    nlptg_model_codellama_13b_python = forms.BooleanField(required=False)
    nlptg_model_codellama_34b = forms.BooleanField(required=False)
    nlptg_model_codellama_34b_instruct = forms.BooleanField(required=False)
    nlptg_model_codellama_34b_python = forms.BooleanField(required=False)

    # Falcom Models
    nlptg_model_falcon_7b = forms.BooleanField(required=False)
    nlptg_model_falcon_7b_instruct = forms.BooleanField(required=False)
    nlptg_model_falcon_40b = forms.BooleanField(required=False)
    nlptg_model_falcon_40b_instruct = forms.BooleanField(required=False)
    nlptg_model_falcon_180b = forms.BooleanField(required=False)
    nlptg_model_falcon_180b_chat = forms.BooleanField(required=False)

    # LLaMa 2 Models
    nlptg_model_llama2_7b = forms.BooleanField(required=False)
    nlptg_model_llama2_7b_chat = forms.BooleanField(required=False)
    nlptg_model_llama2_7b_code = forms.BooleanField(required=False)
    nlptg_model_llama2_13b = forms.BooleanField(required=False)
    nlptg_model_llama2_13b_chat = forms.BooleanField(required=False)
    nlptg_model_llama2_70b = forms.BooleanField(required=False)
    nlptg_model_llama2_70b_chat = forms.BooleanField(required=False)

    # Med42 Model
    nlptg_model_med42_70b = forms.BooleanField(required=False)

    # MedAlpaca Model
    nlptg_model_medalpaca_13b = forms.BooleanField(required=False)

    # Meditron Models
    nlptg_model_meditron_7b = forms.BooleanField(required=False)
    nlptg_model_meditron_7b_chat = forms.BooleanField(required=False)
    nlptg_model_meditron_70b = forms.BooleanField(required=False)

    # Mistral Models
    nlptg_model_mistral_7b = forms.BooleanField(required=False)
    nlptg_model_mistral_7b_instruct = forms.BooleanField(required=False)
    nlptg_model_mistral_7b_openorca = forms.BooleanField(required=False)

    # Mixtral Models
    nlptg_model_mixtral_8x7b = forms.BooleanField(required=False)
    nlptg_model_mixtral_8x7b_instruct = forms.BooleanField(required=False)

    # Neural-Chat Model
    nlptg_model_neuralchat_7b = forms.BooleanField(required=False)

    # OpenChat Model
    nlptg_model_openchat_7b = forms.BooleanField(required=False)

    # OpenLLaMA
    nlptg_model_openllama_3b = forms.BooleanField(required=False)
    nlptg_model_openllama_7b = forms.BooleanField(required=False)
    nlptg_model_openllama_13b = forms.BooleanField(required=False)

    # Orca 2 models
    nlptg_model_orca2_7b = forms.BooleanField(required=False)
    mlptg_model_orca2_13b = forms.BooleanField(required=False)

    # PsyMedRP
    mlptg_model_psymedrp_13b = forms.BooleanField(required=False)
    mlptg_model_psymedrp_20b = forms.BooleanField(required=False)

    # Python Code
    nlptg_model_python_code_13b = forms.BooleanField(required=False)
    mlptg_model_python_code_33b = forms.BooleanField(required=False)

    # Starling LM
    nlptg_model_starlinglm_7b_alpha = forms.BooleanField(required=False)

    # Vicuna
    nlptg_model_vicuna_7b = forms.BooleanField(required=False)
    nlptg_model_vicuna_13b = forms.BooleanField(required=False)
    nlptg_model_vicuna_33b = forms.BooleanField(required=False)
    nlptg_model_vicuna_33b_coder = forms.BooleanField(required=False)

    # Vicuna
    nlptg_model_wizardlm_7b = forms.BooleanField(required=False)
    nlptg_model_wizardlm_13b = forms.BooleanField(required=False)
    nlptg_model_wizardlm_70b = forms.BooleanField(required=False)

    # Zephyr
    nlptg_model_zephyr_3b = forms.BooleanField(required=False)
    nlptg_model_zephyr_7b_alpha = forms.BooleanField(required=False)
    nlptg_model_zephyr_7b_beta = forms.BooleanField(required=False)


class NLPEmotionalAnalysisForm(forms.Form):
    """Class for creating NLP Emotional Analysis form."""

    # Alfred
    nlpema_model_bert = forms.BooleanField(required=False)


class UserRegistrationForm(forms.ModelForm):
    """Class for creating user registration form."""

    password = forms.CharField(label="Password", widget=forms.PasswordInput)
    password2 = forms.CharField(
        label="Repeat Password", widget=forms.PasswordInput,
    )

    class Meta:
        model = User
        fields = ["username", "first_name", "last_name", "email"]

    def clean_password2(self):
        """Validate password"""
        cd = self.cleaned_data
        if cd["password"] != cd["password2"]:
            msg = "Passwords do not match."
            raise forms.ValidationError(msg)
        return cd["password2"]

    def clean_email(self):
        """Validate email"""
        data = self.cleaned_data["email"]
        if User.objects.filter(email=data).exists():
            msg = "Email already registered."
            raise forms.ValidationError(msg)
        return data


class UserEditForm(forms.ModelForm):
    """Class for creating Profile Edition mform."""

    class Meta:
        model = User
        fields = ["first_name", "last_name", "email"]

    def clean_email(self):
        """Validate email"""
        data = self.cleaned_data["email"]
        qs = User.objects.exclude(id=self.instance.id).filter(email=data)
        if qs.exists():
            msg = "Emaii already registered."
            raise forms.ValidationError(msg)
        return data


class ProfileEditForm(forms.ModelForm):
    """Class for creating Profile Edition form"""

    class Meta:
        model = Profile
        fields = ["date_of_birth", "photo"]
