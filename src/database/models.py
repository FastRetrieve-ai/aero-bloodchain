"""
SQLAlchemy models for emergency cases
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Time
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class EmergencyCase(Base):
    """Emergency case model representing all fields from CSV data"""
    
    __tablename__ = "emergency_cases"
    
    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Basic Case Information
    case_number = Column(String(50), unique=True, index=True, nullable=False)  # 序號
    date = Column(DateTime, index=True)  # 日期
    brigade = Column(String(50), index=True)  # 大隊
    team = Column(String(50))  # 分隊
    vehicle_number = Column(String(50))  # 車號
    special_rescue_team = Column(Boolean)  # 專救隊
    
    # Dispatch Information
    dispatch_reason = Column(String(100), index=True)  # 派遣原因
    dispatch_classification = Column(String(20))  # 派遣分類 (BLS/ALS)
    support_dispatch = Column(Boolean)  # 支援派遣
    incident_district = Column(String(50), index=True)  # 發生地點行政區
    receiving_unit = Column(String(100))  # 受案單位
    case_address = Column(Text)  # 案發地址
    actual_address = Column(Text)  # 實際地址
    report_phone = Column(String(20))  # 報案電話
    
    # Non-transport reasons
    no_transport_reason = Column(String(100))  # 未送醫原因
    no_contact_reason = Column(String(100))  # 未接觸原因
    
    # Patient Information
    patient_name = Column(String(100))  # 患者姓名
    patient_id = Column(String(20))  # 患者身份證字號
    patient_weight = Column(Float)  # 患者體重
    patient_birthday = Column(DateTime)  # 患者生日
    patient_age = Column(Integer)  # 患者年齡
    patient_gender = Column(String(10))  # 患者性別
    patient_address = Column(Text)  # 患者住址
    patient_phone = Column(String(20))  # 患者電話
    medical_record_number = Column(String(50))  # 病歷號
    
    # Hospital Information
    destination_hospital = Column(String(100), index=True)  # 後送醫院
    triage_level = Column(String(20), index=True)  # 檢傷分級
    hospital_selection = Column(String(50))  # 醫院選擇
    bed_reserved = Column(Boolean)  # 扣床
    notes = Column(Text)  # 補述欄
    
    # Timeline (all in HH:MM:SS format or seconds)
    dispatch_time = Column(String(20))  # 派遣時間
    departure_time = Column(String(20))  # 出勤時間
    response_time_seconds = Column(Integer)  # 反應時間(秒)
    arrival_time = Column(String(20))  # 抵達時間
    patient_contact_time = Column(String(20))  # 接觸病患時間
    on_scene_time_seconds = Column(Integer)  # 現場處置時間(秒)
    leave_scene_time = Column(String(20))  # 離開現場時間
    transport_time_seconds = Column(Integer)  # 送醫時間(秒)
    arrival_hospital_time = Column(String(20))  # 抵達醫院時間
    hospital_stay_seconds = Column(Integer)  # 留院時間(秒)
    leave_hospital_time = Column(String(20))  # 離開醫院時間
    return_time_seconds = Column(Integer)  # 返隊時間(秒)
    return_station_time = Column(String(20))  # 返隊時間
    
    # Chief Complaint
    chief_complaint = Column(String(200), index=True)  # 傷病主訴群
    chief_complaint_other = Column(Text)  # 其他-其他說明
    trauma_complaint = Column(String(200))  # 創傷導致的主訴群
    
    # Medical History
    past_history = Column(String(200))  # 過去病史
    past_history_detail = Column(Text)  # 過去病史說明
    allergy_history = Column(String(200))  # 過敏史
    allergy_history_other = Column(Text)  # 過敏史其他
    
    # Physical Assessment
    skin_color = Column(String(20))  # 膚色
    left_pupil = Column(String(20))  # 左瞳孔光反射、尺寸(mm)
    right_pupil = Column(String(20))  # 右瞳孔光反射、尺寸(mm)
    trachea_position = Column(String(20))  # 氣管位置
    jugular_vein_distention = Column(String(20))  # 頸靜脈怒張
    blood_glucose_measured = Column(Boolean)  # 血糖測量
    assessment_not_measured_reason = Column(String(100))  # 評估-未測量原因
    assessment_notes = Column(Text)  # 評估說明
    blood_glucose_mgdl = Column(Float)  # mg/dl
    critical_case = Column(Boolean, index=True)  # 危急個案
    pain_score = Column(Integer)  # 疼痛指數
    
    # Treatment - General
    general_treatment = Column(Text)  # 一般處置
    general_treatment_notes = Column(Text)  # 一般處置-說明
    
    # Treatment - Respiratory
    respiratory_treatment = Column(String(100))  # 呼吸處置
    oxygen_nasal_flow = Column(Float)  # 氧氣鼻管流量(L/min)
    oxygen_mask_flow = Column(Float)  # 氧氣面罩流量(L/min)
    nrm_flow = Column(Float)  # NRM流量(L/min)
    positive_pressure_flow = Column(Float)  # 正壓給氧流量(L/min)
    
    # Treatment - Advanced
    advanced_treatment = Column(Text)  # 加護處置
    ntg_tablets = Column(Integer)  # NTG(片)
    glucose_50_ampules = Column(Integer)  # 50%G.W.(支)
    glucose_5_ml = Column(Float)  # 5%G.W.(ml)
    normal_saline_ml = Column(Float)  # 0.9%N/S(ml)
    lactated_ringers_ml = Column(Float)  # L.R(ml)
    cpr_duration_minutes = Column(Integer)  # CPR時間(分鐘)
    aed_shocks = Column(Integer)  # AED電擊次數(次)
    
    # Treatment - Trauma
    trauma_treatment = Column(Text)  # 創傷處置
    
    # Vital Signs (3 measurements)
    consciousness_1 = Column(String(50))
    consciousness_2 = Column(String(50))
    consciousness_3 = Column(String(50))
    gcs_1 = Column(String(20))
    gcs_2 = Column(String(20))
    gcs_3 = Column(String(20))
    respiratory_rate_1 = Column(String(50))
    respiratory_rate_2 = Column(String(50))
    respiratory_rate_3 = Column(String(50))
    pulse_1 = Column(String(50))
    pulse_2 = Column(String(50))
    pulse_3 = Column(String(50))
    blood_pressure_1 = Column(String(50))
    blood_pressure_2 = Column(String(50))
    blood_pressure_3 = Column(String(50))
    temperature_1 = Column(Float)
    temperature_2 = Column(Float)
    temperature_3 = Column(Float)
    spo2_1 = Column(Integer)
    spo2_2 = Column(Integer)
    spo2_3 = Column(Integer)
    etco2_1 = Column(Integer)
    etco2_2 = Column(Integer)
    etco2_3 = Column(Integer)
    
    # Traffic Accident Information
    traffic_accident_patient = Column(Boolean)  # 交通事故患者
    traffic_accident_type = Column(String(100))  # 交通事故種類
    traffic_accident_other = Column(Text)  # 交通事故種類-其他說明
    protective_device_status = Column(String(100))  # 保護裝置狀態
    non_traffic_accident_type = Column(String(100))  # 非交通事故種類
    fall_height = Column(Float)  # 高度
    fall_height_other = Column(Text)  # 高度-其他說明
    cervical_spine_fixation = Column(String(50))  # 頸椎固定
    major_trauma_basis = Column(Text)  # 重大外傷依據
    
    # Location Information
    patient_location = Column(String(100))  # 患者倒地所在場所
    patient_location_other = Column(Text)  # 患者倒地所在場所-其他說明
    down_time_before_contact = Column(String(50))  # 接觸前倒地時間
    
    # CPR Information
    bystander_cpr_before_contact = Column(Boolean)  # 接觸前是否已有旁觀者施行CPR
    cpr_content = Column(Text)  # 實施CPR內容
    chest_compression_quality = Column(String(50))  # 胸外按摩品質
    bystander_cpr_trained = Column(String(50))  # 旁觀者曾接受CPR訓練
    last_training = Column(String(50))  # 最後訓練
    bystander_pad_used = Column(Boolean)  # 旁觀者有無使用PAD(公眾AED)
    pad_shock_count = Column(Integer)  # 電擊次數
    auto_cpr_machine = Column(Boolean)  # 使用自動心肺復甦機
    treatment_notes = Column(Text)  # 處置-說明
    
    # Cardiac Information
    pre_hospital_rhythm = Column(String(50))  # 到院前患者心律
    shock_times = Column(Text)  # 電擊次數(時間)
    aed_shock_count_detail = Column(Integer)  # AED電擊次數
    pre_hospital_rosc_time = Column(String(20))  # 到院前恢復心跳時間
    cardiac_arrest_cause = Column(String(100))  # 心搏停止原因臆測
    
    # Airway Management
    airway_treatment = Column(String(100))  # 氣道處置
    airway_not_performed_reason = Column(String(100))  # 氣道處置未執行原因
    lma_size = Column(String(20))  # LMA尺寸
    endo_size = Column(String(20))  # ENDO尺寸
    ett_position_confirmation = Column(String(100))  # 氣管內管位置確認
    bilateral_chest = Column(String(50))  # 二側胸部
    edd = Column(String(50))  # EDD
    mechanical_ventilator = Column(Boolean)  # 使用機械式給氧裝置
    
    # Medications
    epinephrine_count = Column(Integer)  # 給予Epinephrine數量
    epinephrine_not_given_reason = Column(String(100))  # 未施打Epinephrine原因
    amiodarone_count = Column(Integer)  # 給予Amiodarone數量
    amiodarone_not_given_reason = Column(String(100))  # 未施打Amiodarone原因
    amiodarone_not_given_detail = Column(Text)  # 未施打Amiodarone原因-說明
    
    # Pediatric Assessment
    pediatric_appearance = Column(String(50))  # 小兒評估三角-外觀
    pediatric_breathing = Column(String(50))  # 小兒評估三角-呼吸
    pediatric_circulation = Column(String(50))  # 小兒評估三角-循環
    
    # Stroke Assessment
    stroke_test = Column(String(100))  # 腦中風測試
    stroke_assessment_other = Column(Text)  # 腦中風評估其他
    last_normal_time = Column(String(20))  # 最後正常時間(24hr制)
    symptom_onset = Column(String(50))  # 症狀發生
    assessment_factors = Column(Text)  # 評估因子
    
    # Allergic Reaction
    severe_allergic_reaction = Column(Boolean)  # 患者出現嚴重過敏反應
    epinephrine_03mg_im = Column(Integer)  # Epinephrine 0.3mg  IM(支)
    spo2_value = Column(Integer)  # SPO2
    accessory_muscle_use = Column(Boolean)  # 使用呼吸輔助肌
    abnormal_breathing = Column(String(100))  # 呼吸異常
    
    # Bronchodilator Treatment
    age_range_bronchodilator = Column(String(50))  # 年齡範圍(Berotec/Albuterol給藥)
    berotec_given = Column(Boolean)  # 給予Berotec支氣管擴張劑 2puff
    albuterol_nebulizer = Column(Boolean)  # Albuterol 5mg+0.9%N/S 4c.c噴霧治療
    
    # Seizure
    status_epilepticus = Column(Boolean)  # 癲癇重積
    status_epilepticus_other = Column(Text)  # 癲癇重積-其他說明
    midazolam_given = Column(Boolean)  # Midazolam給藥
    medication_not_given_reason = Column(String(100))  # 未給藥原因
    medication_time = Column(String(20))  # 給藥時間
    symptom_start_time = Column(String(20))  # 症狀開始時間
    
    # Cardiac Assessment
    aortic_dissection_pulse_diff = Column(Boolean)  # 主動脈剝離，測量脈搏壓差≧10mmHg
    rales_or_tearing_pain = Column(Boolean)  # 濕囉音或撕裂性胸痛或背痛
    ntg_time = Column(String(20))  # 舌下NTG  時間(若SBP<90mmHg不給藥)
    sbp = Column(Integer)  # SBP
    antiplatelet_contraindication = Column(Boolean)  # 血小板凝集抑制劑使用有無禁忌症
    contraindication = Column(Text)  # 禁忌症
    aspirin_time = Column(String(20))  # 口服Aspirin 300mg 時間
    ticagrelor_time = Column(String(20))  # 口服Ticagrelor 180mg 時間
    atropine_time = Column(String(20))  # Atropine 0.5mg IV 時間
    rhythm_interpretation = Column(String(50))  # 心律臆斷
    not_used_reason = Column(String(100))  # 未使用原因
    tcp_used = Column(Boolean)  # 使用TCP
    output_current = Column(String(50))  # 輸出電流
    rhythm = Column(String(50))  # 心律
    time = Column(String(20))  # 時間
    
    # EMT Information
    emt1_level = Column(String(20))  # EMT1等級
    emt2_level = Column(String(20))  # EMT2等級
    emt3_level = Column(String(20))  # EMT3等級
    emt4_level = Column(String(20))  # EMT4等級
    emt1_name = Column(String(50))  # EMT1姓名
    emt2_name = Column(String(50))  # EMT2姓名
    emt3_name = Column(String(50))  # EMT3姓名
    emt4_name = Column(String(50))  # EMT4姓名
    cooperation_role1 = Column(String(50))  # 協同角色1
    cooperation_role2 = Column(String(50))  # 協同角色2
    cooperation_role3 = Column(String(50))  # 協同角色3
    
    # Documentation
    disinfection_level = Column(String(50))  # 消毒等級
    burn_diagram = Column(Boolean)  # 燒燙傷圖
    triage_diagram = Column(Boolean)  # 檢傷圖
    ecg_diagram = Column(Boolean)  # 心電圖
    
    # Medication Assessment
    ntg_assessment = Column(Text)  # NTG(舌下含片）使用評估
    aspirin_assessment = Column(Text)  # ASPIRIN使用評估
    ticagrelor_assessment = Column(Text)  # Ticagrelor使用評估（有任一情形不能給藥）
    atropine_assessment = Column(Text)  # Atropine使用評估
    transamin_assessment = Column(Text)  # Transamin用藥評估
    transamin_given = Column(Boolean)  # Transamin 1000mg＋N/S 100ml IVD 10mins(3滴/秒)
    nahco3_assessment = Column(Text)  # NaHCO3用藥評估
    nahco3_given = Column(Boolean)  # 4Amp NaHCO3+ N/S 1000ml drip 1hr IVD(5滴/秒)
    
    # OHCA Information
    witnessed_ohca = Column(Boolean)  # 是否目擊OHCA
    witness_personnel = Column(String(100))  # 目擊OHCA人員
    bystander_cpr_non_emt = Column(Boolean)  # 是否旁觀者CPR(非EMT)
    cpr_performed = Column(Boolean)  # 是否CPR
    cpr_duration = Column(Integer)  # 時間（分鐘）
    cpr_not_performed_reason = Column(String(100))  # 未執行CPR原因
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<EmergencyCase(case_number={self.case_number}, date={self.date}, dispatch_reason={self.dispatch_reason})>"

