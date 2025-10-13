"""
Data loader script to import emergency cases from CSV/Excel files into the database
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import DatabaseManager
from src.database.models import EmergencyCase
from src.config import DATA_DIR


def parse_datetime(date_str: Any) -> datetime:
    """Parse datetime from various formats"""
    if pd.isna(date_str):
        return None
    
    if isinstance(date_str, datetime):
        return date_str
    
    try:
        # Try different date formats
        for fmt in ['%Y/%m/%d', '%Y-%m-%d', '%Y/%m/%d %H:%M:%S']:
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        return None
    except:
        return None


def parse_boolean(value: Any) -> bool:
    """Parse boolean values"""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ['是', 'yes', 'true', '1']
    return bool(value)


def parse_float(value: Any) -> float:
    """Safely parse float values"""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except:
        return None


def parse_int(value: Any) -> int:
    """Safely parse integer values"""
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except:
        return None


def clean_string(value: Any) -> str:
    """Clean string values"""
    if pd.isna(value):
        return None
    return str(value).strip()


def row_to_case_dict(row: pd.Series) -> Dict[str, Any]:
    """Convert a DataFrame row to EmergencyCase dictionary"""
    return {
        'case_number': clean_string(row.get('序號')),
        'date': parse_datetime(row.get('日期')),
        'brigade': clean_string(row.get('大隊')),
        'team': clean_string(row.get('分隊')),
        'vehicle_number': clean_string(row.get('車號')),
        'special_rescue_team': parse_boolean(row.get('專救隊')),
        'dispatch_reason': clean_string(row.get('派遣原因')),
        'dispatch_classification': clean_string(row.get('派遣分類')),
        'support_dispatch': parse_boolean(row.get('支援派遣')),
        'incident_district': clean_string(row.get('發生地點行政區')),
        'receiving_unit': clean_string(row.get('受案單位')),
        'case_address': clean_string(row.get('案發地址')),
        'actual_address': clean_string(row.get('實際地址')),
        'report_phone': clean_string(row.get('報案電話')),
        'no_transport_reason': clean_string(row.get('未送醫原因')),
        'no_contact_reason': clean_string(row.get('未接觸原因')),
        'patient_name': clean_string(row.get('患者姓名')),
        'patient_id': clean_string(row.get('患者身份證字號')),
        'patient_weight': parse_float(row.get('患者體重')),
        'patient_birthday': parse_datetime(row.get('患者生日')),
        'patient_age': parse_int(row.get('患者年齡')),
        'patient_gender': clean_string(row.get('患者性別')),
        'patient_address': clean_string(row.get('患者住址')),
        'patient_phone': clean_string(row.get('患者電話')),
        'medical_record_number': clean_string(row.get('病歷號')),
        'destination_hospital': clean_string(row.get('後送醫院')),
        'triage_level': clean_string(row.get('檢傷分級')),
        'hospital_selection': clean_string(row.get('醫院選擇')),
        'bed_reserved': parse_boolean(row.get('扣床')),
        'notes': clean_string(row.get('補述欄')),
        'dispatch_time': clean_string(row.get('派遣時間')),
        'departure_time': clean_string(row.get('出勤時間')),
        'response_time_seconds': parse_int(row.get('反應時間(秒)')),
        'arrival_time': clean_string(row.get('抵達時間')),
        'patient_contact_time': clean_string(row.get('接觸病患時間')),
        'on_scene_time_seconds': parse_int(row.get('現場處置時間(秒)')),
        'leave_scene_time': clean_string(row.get('離開現場時間')),
        'transport_time_seconds': parse_int(row.get('送醫時間(秒)')),
        'arrival_hospital_time': clean_string(row.get('抵達醫院時間')),
        'hospital_stay_seconds': parse_int(row.get('留院時間(秒)')),
        'leave_hospital_time': clean_string(row.get('離開醫院時間')),
        'return_time_seconds': parse_int(row.get('返隊時間(秒)')),
        'return_station_time': clean_string(row.get('返隊時間')),
        'chief_complaint': clean_string(row.get('傷病主訴群')),
        'chief_complaint_other': clean_string(row.get('其他-其他說明')),
        'trauma_complaint': clean_string(row.get('創傷導致的主訴群')),
        'past_history': clean_string(row.get('過去病史')),
        'past_history_detail': clean_string(row.get('過去病史說明')),
        'allergy_history': clean_string(row.get('過敏史')),
        'allergy_history_other': clean_string(row.get('過敏史其他')),
        'skin_color': clean_string(row.get('膚色')),
        'left_pupil': clean_string(row.get('左瞳孔光反射、尺寸(mm)')),
        'right_pupil': clean_string(row.get('右瞳孔光反射、尺寸(mm)')),
        'trachea_position': clean_string(row.get('氣管位置')),
        'jugular_vein_distention': clean_string(row.get('頸靜脈怒張')),
        'blood_glucose_measured': parse_boolean(row.get('血糖測量')),
        'assessment_not_measured_reason': clean_string(row.get('評估-未測量原因')),
        'assessment_notes': clean_string(row.get('評估說明')),
        'blood_glucose_mgdl': parse_float(row.get('mg/dl')),
        'critical_case': parse_boolean(row.get('危急個案')),
        'pain_score': parse_int(row.get('疼痛指數')),
        'general_treatment': clean_string(row.get('一般處置')),
        'general_treatment_notes': clean_string(row.get('一般處置-說明')),
        'respiratory_treatment': clean_string(row.get('呼吸處置')),
        'oxygen_nasal_flow': parse_float(row.get('氧氣鼻管流量(L/min)')),
        'oxygen_mask_flow': parse_float(row.get('氧氣面罩流量(L/min)')),
        'nrm_flow': parse_float(row.get('NRM流量(L/min)')),
        'positive_pressure_flow': parse_float(row.get('正壓給氧流量(L/min)')),
        'advanced_treatment': clean_string(row.get('加護處置')),
        'ntg_tablets': parse_int(row.get('NTG(片)')),
        'glucose_50_ampules': parse_int(row.get('50%G.W.(支)')),
        'glucose_5_ml': parse_float(row.get('5%G.W.(ml)')),
        'normal_saline_ml': parse_float(row.get('0.9%N/S(ml)')),
        'lactated_ringers_ml': parse_float(row.get('L.R(ml)')),
        'cpr_duration_minutes': parse_int(row.get('CPR時間(分鐘)')),
        'aed_shocks': parse_int(row.get('AED電擊次數(次)')),
        'trauma_treatment': clean_string(row.get('創傷處置')),
        'consciousness_1': clean_string(row.get('意識狀態1')),
        'consciousness_2': clean_string(row.get('意識狀態2')),
        'consciousness_3': clean_string(row.get('意識狀態3')),
        'gcs_1': clean_string(row.get('GCS1')),
        'gcs_2': clean_string(row.get('GCS2')),
        'gcs_3': clean_string(row.get('GCS3')),
        'respiratory_rate_1': clean_string(row.get('呼吸1')),
        'respiratory_rate_2': clean_string(row.get('呼吸2')),
        'respiratory_rate_3': clean_string(row.get('呼吸3')),
        'pulse_1': clean_string(row.get('脈搏1')),
        'pulse_2': clean_string(row.get('脈搏2')),
        'pulse_3': clean_string(row.get('脈搏3')),
        'blood_pressure_1': clean_string(row.get('血壓1')),
        'blood_pressure_2': clean_string(row.get('血壓2')),
        'blood_pressure_3': clean_string(row.get('血壓3')),
        'temperature_1': parse_float(row.get('體溫1')),
        'temperature_2': parse_float(row.get('體溫2')),
        'temperature_3': parse_float(row.get('體溫3')),
        'spo2_1': parse_int(row.get('血氧1')),
        'spo2_2': parse_int(row.get('血氧2')),
        'spo2_3': parse_int(row.get('血氧3')),
        'etco2_1': parse_int(row.get('EtCO2_1')),
        'etco2_2': parse_int(row.get('EtCO2_2')),
        'etco2_3': parse_int(row.get('EtCO2_3')),
        'traffic_accident_patient': parse_boolean(row.get('交通事故患者')),
        'traffic_accident_type': clean_string(row.get('交通事故種類')),
        'traffic_accident_other': clean_string(row.get('交通事故種類-其他說明')),
        'protective_device_status': clean_string(row.get('保護裝置狀態')),
        'non_traffic_accident_type': clean_string(row.get('非交通事故種類')),
        'fall_height': parse_float(row.get('高度')),
        'fall_height_other': clean_string(row.get('高度-其他說明')),
        'cervical_spine_fixation': clean_string(row.get('頸椎固定')),
        'major_trauma_basis': clean_string(row.get('重大外傷依據')),
        'patient_location': clean_string(row.get('患者倒地所在場所')),
        'patient_location_other': clean_string(row.get('患者倒地所在場所-其他說明')),
        'down_time_before_contact': clean_string(row.get('接觸前倒地時間')),
        'bystander_cpr_before_contact': parse_boolean(row.get('接觸前是否已有旁觀者施行CPR')),
        'cpr_content': clean_string(row.get('實施CPR內容')),
        'chest_compression_quality': clean_string(row.get('胸外按摩品質')),
        'bystander_cpr_trained': clean_string(row.get('旁觀者曾接受CPR訓練')),
        'last_training': clean_string(row.get('最後訓練')),
        'bystander_pad_used': parse_boolean(row.get('旁觀者有無使用PAD(公眾AED)')),
        'pad_shock_count': parse_int(row.get('電擊次數')),
        'auto_cpr_machine': parse_boolean(row.get('使用自動心肺復甦機')),
        'treatment_notes': clean_string(row.get('處置-說明')),
        'pre_hospital_rhythm': clean_string(row.get('到院前患者心律')),
        'shock_times': clean_string(row.get('電擊次數(時間)')),
        'aed_shock_count_detail': parse_int(row.get('AED電擊次數')),
        'pre_hospital_rosc_time': clean_string(row.get('到院前恢復心跳時間')),
        'cardiac_arrest_cause': clean_string(row.get('心搏停止原因臆測')),
        'airway_treatment': clean_string(row.get('氣道處置')),
        'airway_not_performed_reason': clean_string(row.get('氣道處置未執行原因')),
        'lma_size': clean_string(row.get('LMA尺寸')),
        'endo_size': clean_string(row.get('ENDO尺寸')),
        'ett_position_confirmation': clean_string(row.get('氣管內管位置確認')),
        'bilateral_chest': clean_string(row.get('二側胸部')),
        'edd': clean_string(row.get('EDD')),
        'mechanical_ventilator': parse_boolean(row.get('使用機械式給氧裝置')),
        'epinephrine_count': parse_int(row.get('給予Epinephrine數量')),
        'epinephrine_not_given_reason': clean_string(row.get('未施打Epinephrine原因')),
        'amiodarone_count': parse_int(row.get('給予Amiodarone數量')),
        'amiodarone_not_given_reason': clean_string(row.get('未施打Amiodarone原因')),
        'amiodarone_not_given_detail': clean_string(row.get('未施打Amiodarone原因-說明')),
        'pediatric_appearance': clean_string(row.get('小兒評估三角-外觀')),
        'pediatric_breathing': clean_string(row.get('小兒評估三角-呼吸')),
        'pediatric_circulation': clean_string(row.get('小兒評估三角-循環')),
        'stroke_test': clean_string(row.get('腦中風測試')),
        'stroke_assessment_other': clean_string(row.get('腦中風評估其他')),
        'last_normal_time': clean_string(row.get('最後正常時間(24hr制)')),
        'symptom_onset': clean_string(row.get('症狀發生')),
        'assessment_factors': clean_string(row.get('評估因子')),
        'severe_allergic_reaction': parse_boolean(row.get('患者出現嚴重過敏反應')),
        'epinephrine_03mg_im': parse_int(row.get('Epinephrine 0.3mg  IM(支)')),
        'spo2_value': parse_int(row.get('SPO2')),
        'accessory_muscle_use': parse_boolean(row.get('使用呼吸輔助肌')),
        'abnormal_breathing': clean_string(row.get('呼吸異常')),
        'age_range_bronchodilator': clean_string(row.get('年齡範圍(Berotec/Albuterol給藥)')),
        'berotec_given': parse_boolean(row.get('給予Berotec支氣管擴張劑 2puff')),
        'albuterol_nebulizer': parse_boolean(row.get('Albuterol 5mg+0.9%N/S 4c.c噴霧治療')),
        'status_epilepticus': parse_boolean(row.get('癲癇重積')),
        'status_epilepticus_other': clean_string(row.get('癲癇重積-其他說明')),
        'midazolam_given': parse_boolean(row.get('Midazolam給藥')),
        'medication_not_given_reason': clean_string(row.get('未給藥原因')),
        'medication_time': clean_string(row.get('給藥時間')),
        'symptom_start_time': clean_string(row.get('症狀開始時間')),
        'aortic_dissection_pulse_diff': parse_boolean(row.get('主動脈剝離，測量脈搏壓差≧10mmHg')),
        'rales_or_tearing_pain': parse_boolean(row.get('濕囉音或撕裂性胸痛或背痛')),
        'ntg_time': clean_string(row.get('舌下NTG  時間(若SBP<90mmHg不給藥)')),
        'sbp': parse_int(row.get('SBP')),
        'antiplatelet_contraindication': parse_boolean(row.get('血小板凝集抑制劑使用有無禁忌症')),
        'contraindication': clean_string(row.get('禁忌症')),
        'aspirin_time': clean_string(row.get('口服Aspirin 300mg 時間')),
        'ticagrelor_time': clean_string(row.get('口服Ticagrelor 180mg 時間')),
        'atropine_time': clean_string(row.get('Atropine 0.5mg IV 時間')),
        'rhythm_interpretation': clean_string(row.get('心律臆斷')),
        'not_used_reason': clean_string(row.get('未使用原因')),
        'tcp_used': parse_boolean(row.get('使用TCP')),
        'output_current': clean_string(row.get('輸出電流')),
        'rhythm': clean_string(row.get('心律')),
        'time': clean_string(row.get('時間')),
        'emt1_level': clean_string(row.get('EMT1等級')),
        'emt2_level': clean_string(row.get('EMT2等級')),
        'emt3_level': clean_string(row.get('EMT3等級')),
        'emt4_level': clean_string(row.get('EMT4等級')),
        'emt1_name': clean_string(row.get('EMT1姓名')),
        'emt2_name': clean_string(row.get('EMT2姓名')),
        'emt3_name': clean_string(row.get('EMT3姓名')),
        'emt4_name': clean_string(row.get('EMT4姓名')),
        'cooperation_role1': clean_string(row.get('協同角色1')),
        'cooperation_role2': clean_string(row.get('協同角色2')),
        'cooperation_role3': clean_string(row.get('協同角色3')),
        'disinfection_level': clean_string(row.get('消毒等級')),
        'burn_diagram': parse_boolean(row.get('燒燙傷圖')),
        'triage_diagram': parse_boolean(row.get('檢傷圖')),
        'ecg_diagram': parse_boolean(row.get('心電圖')),
        'ntg_assessment': clean_string(row.get('NTG(舌下含片）使用評估')),
        'aspirin_assessment': clean_string(row.get('ASPIRIN使用評估')),
        'ticagrelor_assessment': clean_string(row.get('Ticagrelor使用評估（有任一情形不能給藥）')),
        'atropine_assessment': clean_string(row.get('Atropine使用評估')),
        'transamin_assessment': clean_string(row.get('Transamin用藥評估')),
        'transamin_given': parse_boolean(row.get('Transamin 1000mg＋N/S 100ml IVD 10mins(3滴/秒)')),
        'nahco3_assessment': clean_string(row.get('NaHCO3用藥評估')),
        'nahco3_given': parse_boolean(row.get('4Amp NaHCO3+ N/S 1000ml drip 1hr IVD(5滴/秒)')),
        'witnessed_ohca': parse_boolean(row.get('是否目擊OHCA')),
        'witness_personnel': clean_string(row.get('目擊OHCA人員')),
        'bystander_cpr_non_emt': parse_boolean(row.get('是否旁觀者CPR(非EMT)')),
        'cpr_performed': parse_boolean(row.get('是否CPR')),
        'cpr_duration': parse_int(row.get('時間（分鐘）')),
        'cpr_not_performed_reason': clean_string(row.get('未執行CPR原因')),
    }


def load_file(file_path: Path, db_manager: DatabaseManager) -> int:
    """Load data from a CSV or Excel file"""
    print(f"Loading file: {file_path}")
    
    # Read file based on extension
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        print(f"Unsupported file format: {file_path.suffix}")
        return 0
    
    print(f"Found {len(df)} records in file")
    
    # Load each row
    loaded_count = 0
    skipped_count = 0
    
    session = db_manager.get_session()
    
    for idx, row in df.iterrows():
        try:
            case_data = row_to_case_dict(row)
            
            # Skip if case_number already exists
            existing = session.query(EmergencyCase).filter_by(
                case_number=case_data['case_number']
            ).first()
            
            if existing:
                skipped_count += 1
                continue
            
            case = EmergencyCase(**case_data)
            session.add(case)
            loaded_count += 1
            
            # Commit every 100 records
            if loaded_count % 100 == 0:
                session.commit()
                print(f"Loaded {loaded_count} records...")
                
        except Exception as e:
            print(f"Error loading row {idx}: {e}")
            session.rollback()
            continue
    
    # Final commit
    session.commit()
    session.close()
    
    print(f"Successfully loaded {loaded_count} records, skipped {skipped_count} duplicates")
    return loaded_count


def main():
    """Main function to load all data files"""
    print("Starting data load...")
    
    # Initialize database
    db_manager = DatabaseManager()
    db_manager.create_tables()
    print("Database tables created")
    
    # Find all CSV and Excel files in data directory
    data_files = []
    for pattern in ['*.csv', '*.xlsx', '*.xls']:
        data_files.extend(DATA_DIR.glob(pattern))
    
    if not data_files:
        print(f"No data files found in {DATA_DIR}")
        return
    
    print(f"Found {len(data_files)} data files")
    
    total_loaded = 0
    for file_path in data_files:
        loaded = load_file(file_path, db_manager)
        total_loaded += loaded
    
    print(f"\n{'='*50}")
    print(f"Data loading complete!")
    print(f"Total records loaded: {total_loaded}")
    print(f"{'='*50}")
    
    # Display statistics
    stats = db_manager.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"Total cases: {stats['total_cases']}")
    print(f"Critical cases: {stats['critical_cases']}")
    print(f"Average response time: {stats['avg_response_time_seconds']:.1f} seconds")


if __name__ == "__main__":
    main()

