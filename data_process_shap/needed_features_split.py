'''
22.5.17根据处理缺失值后的确定的特征名, 分割出需要用到的所有特征.
PCA和其它不迁移的降维方法使用这个特征'''
import numpy as np
import pandas as pd
important_column = [
    # 一般情况
    '年龄','身高', '体重', 'BMI', 'O型', 'A型' ,'B型', 'AB型' , '乙肝携带' ,
    # 术中情况
    '术式(经典1背驮0)', '手术时间min' ,'无肝期时间min' ,'热缺血时间min' ,'冷缺血时间min', '红细胞', '血浆', '自体血',
#  '术式(经典1背驮2)', '手术时间min' ,'无肝期时间min' ,'热缺血时间min' ,'冷缺血时间min', '红细胞', '血浆', '自体血',
    '4%白蛋白' ,'2%白蛋白', '纯白蛋白g', 'NS' ,'LR', '万汶' ,'佳乐施', '总入量', '出血量', '胸水' ,'腹水', '总尿量',
    # 'I期尿量', 'II期尿量' ,'III期尿量', 
    '速尿mg', '甘露醇ml', '碳酸氢钠ml', '纤维蛋白原g' ,'凝血酶原复合物U',
    'VII因子' ,'氨甲环酸g/h', '氨甲环酸入壶g' ,'去甲肾上腺素维持' ,'去甲肾上腺素出室', '肾上腺素维持', '肾上腺素出室',
    '多巴胺维持mg/h' ,'多巴胺出室' ,'开放时阿托品' ,'开放时最低心率' ,'开放时最低SBP' ,'开放时最低DBP', '开放时最低MBP',
    '再灌注后综合征', '切脾', '肝肾联合移植' ,'特利加压素ml/h',
    # 血常规
    'Hb-pre' ,'HCT-pre' ,'MCV-pre' ,'MCH-pre' ,'MCHC-pre' ,'RDW-CVO-pre' ,'PLT-pre' ,'MPV-pre' ,'PDW-pre' ,'LCR-pre' ,'Hb-post' ,'HCT-post' ,'MCV-post' ,'MCH-post' ,'MCHC-post' ,'RDW-CVO-post' ,'PLT-post' ,'MPV-post' ,'PDW-post' ,'LCR-post' ,'Hb-1' ,'HCT-1' ,'MCV-1' ,'MCH-1' ,'MCHC-1' ,'RDW-CVO-1' ,'PLT-1' ,'MPV-1' ,'PDW-1' ,'LCR-1' ,'Hb-2' ,'HCT-2' ,'MCV-2' ,'MCH-2' ,'MCHC-2' ,'RDW-CVO-2' ,'PLT-2' ,'MPV-2' ,'PDW-2' ,'LCR-2' ,'Hb-3' ,'HCT-3' ,'MCV-3' ,'MCH-3' ,'MCHC-3' ,'RDW-CVO-3' ,'PLT-3' ,'MPV-3' ,'PDW-3' ,'LCR-3' ,'Hb-4' ,'HCT-4' ,'MCV-4' ,'MCH-4' ,'MCHC-4' ,'RDW-CVO-4' ,'PLT-4' ,'MPV-4' ,'PDW-4' ,'LCR-4' ,'Hb-5' ,'HCT-5' ,'MCV-5' ,'MCH-5' ,'MCHC-5' ,'RDW-CVO-5' ,'PLT-5' ,'MPV-5' ,'PDW-5' ,'LCR-5' ,'Hb-6' ,'HCT-6' ,'MCV-6' ,'MCH-6' ,'MCHC-6' ,'RDW-CVO-6' ,'PLT-6' ,'MPV-6' ,'PDW-6' ,'LCR-6' ,'Hb-7' ,'HCT-7' ,'MCV-7' ,'MCH-7' ,'MCHC-7' ,'RDW-CVO-7' ,'PLT-7' ,'MPV-7' ,'PDW-7' ,'LCR-7' ,'Hb-14' ,'HCT-14' ,'MCV-14' ,'MCH-14' ,'MCHC-14' ,'RDW-CVO-14' ,'PLT-14' ,'MPV-14' ,'PDW-14' ,'LCR-14',
    # 生化
    'AST-pre' ,'ALT-pre' ,'TBIL-pre' ,'ALB-pre' ,'BUN-pre' ,'Cr-pre' ,'Glu-pre' ,'K-pre' ,'Na-pre' ,'Ca-pre' ,'AST-post' ,'ALT-post' ,'TBIL-post' ,'ALB-post' ,'BUN-post' ,'Cr-post' ,'K-post' ,'Na-post' ,'Ca-post' ,'AST-1' ,'ALT-1' ,'TBIL-1' ,'ALB-1' ,'BUN-1' ,'Cr-1' ,'K-1' ,'Na-1' ,'Ca-1' ,'AST-2' ,'ALT-2' ,'TBIL-2' ,'ALB-2' ,'BUN-2' ,'Cr-2' ,'K-2' ,'Na-2' ,'Ca-2' ,'AST-3' ,'ALT-3' ,'TBIL-3' ,'ALB-3' ,'BUN-3' ,'Cr-3' ,'K-3' ,'Na-3' ,'Ca-3' ,'AST-4' ,'ALT-4' ,'TBIL-4' ,'ALB-4' ,'BUN-4' ,'Cr-4' ,'K-4' ,'Na-4' ,'Ca-4' ,'AST-5' ,'ALT-5' ,'TBIL-5' ,'ALB-5' ,'BUN-5' ,'Cr-5' ,'K-5' ,'Na-5' ,'Ca-5' ,'AST-6' ,'ALT-6' ,'TBIL-6' ,'ALB-6' ,'BUN-6' ,'Cr-6' ,'K-6' ,'Na-6' ,'Ca-6' ,'AST-7' ,'ALT-7' ,'TBIL-7' ,'ALB-7' ,'BUN-7' ,'Cr-7' ,'K-7' ,'Na-7' ,'Ca-7' ,'AST-14' ,'ALT-14' ,'TBIL-14' ,'ALB-14' ,'BUN-14' ,'Cr-14' ,'K-14' ,'Na-14' ,'Ca-14',
    # 血气
    'HCT-pre','PH-pre' ,'PCO2-pre' ,'PO2-pre' ,'Na-pre.1' ,'K-pre.1' ,'Ca-pre.1' ,'Glu-pre.1' ,'Lac-pre' ,'BE(B)-pre' ,'Hb-pre.1' ,'PH-0' ,'PCO2-0' ,'PO2-0' ,'Na-0' ,'K-0' ,'Ca-0' ,'Glu-0' ,'Lac-0' ,'Hct-0' ,'BE(B)-0' ,'Hb-0' ,'PH-30' ,'PCO2-30' ,'PO2-30' ,'Na-30' ,'K-30' ,'Ca-30' ,'Glu-30' ,'Lac-30' ,'Hct-30' ,'BE(B)-30' ,'Hb-30' ,'PH-60' ,'PCO2-60' ,'PO2-60' ,'Na-60' ,'K-60' ,'Ca-60' ,'Glu-60' ,'Lac-60' ,'Hct-60' ,'BE(B)-60' ,'Hb-60' ,'PH-150' ,'PCO2-150' ,'PO2-150' ,'Na-150' ,'K-150' ,'Ca-150' ,'Glu-150' ,'Lac-150' ,'PH-end' ,'PCO2-end' ,'PO2-end' ,'Na-end' ,'K-end' ,'Ca-end' ,'Glu-end' ,'Lac-end' ,'Hct-end' ,'BE(B)-end' ,'Hb-end' ,'PH-icu' ,'PCO2-icu' ,'PO2-icu' ,'Na-icu' ,'K-icu' ,'Ca-icu' ,'Glu-icu' ,'Lac-icu' ,'Hct-icu' ,'BE(B)-icu' ,'Hb-icu' ,'PH-1d' ,'PCO2-1d' ,'PO2-1d' ,'Na-1d' ,'K-1d' ,'Ca-1d' ,'Glu-1d' ,'Lac-1d' ,'Hct-1d' ,'BE(B)-1d' ,'Hb-1d' ,'PH-2d' ,'PCO2-2d' ,'PO2-2d' ,'Na-2d' ,'K-2d' ,'Ca-2d' ,'Glu-2d' ,'Lac-2d' ,'Hct-2d' ,'BE(B)-2d' ,'Hb-2d',
    # 'PH-pre' ,'PCO2-pre' ,'PO2-pre' ,'Na-pre.1' ,'K-pre.1' ,'Ca-pre.1' ,'Glu-pre.1' ,'Lac-pre' ,'BE(B)-pre' ,'Hb-pre.1' ,'PH-0' ,'PCO2-0' ,'PO2-0' ,'Na-0' ,'K-0' ,'Ca-0' ,'Glu-0' ,'Lac-0' ,'Hct-0' ,'BE(B)-0' ,'Hb-0' ,'PH-30' ,'PCO2-30' ,'PO2-30' ,'Na-30' ,'K-30' ,'Ca-30' ,'Glu-30' ,'Lac-30' ,'Hct-30' ,'BE(B)-30' ,'Hb-30' ,'PH-60' ,'PCO2-60' ,'PO2-60' ,'Na-60' ,'K-60' ,'Ca-60' ,'Glu-60' ,'Lac-60' ,'Hct-60' ,'BE(B)-60' ,'Hb-60' ,'PH-150' ,'PCO2-150' ,'PO2-150' ,'Na-150' ,'K-150' ,'Ca-150' ,'Glu-150' ,'Lac-150' ,'PH-end' ,'PCO2-end' ,'PO2-end' ,'Na-end' ,'K-end' ,'Ca-end' ,'Glu-end' ,'Lac-end' ,'Hct-end' ,'BE(B)-end' ,'Hb-end' ,'PH-icu' ,'PCO2-icu' ,'PO2-icu' ,'Na-icu' ,'K-icu' ,'Ca-icu' ,'Glu-icu' ,'Lac-icu' ,'Hct-icu' ,'BE(B)-icu' ,'Hb-icu' ,'PH-1d' ,'PCO2-1d' ,'PO2-1d' ,'Na-1d' ,'K-1d' ,'Ca-1d' ,'Glu-1d' ,'Lac-1d' ,'Hct-1d' ,'BE(B)-1d' ,'Hb-1d' ,'PH-2d' ,'PCO2-2d' ,'PO2-2d' ,'Na-2d' ,'K-2d' ,'Ca-2d' ,'Glu-2d' ,'Lac-2d' ,'Hct-2d' ,'BE(B)-2d' ,'Hb-2d',
    # 凝血
    'PA-pre' ,'PT-pre' ,'PR-pre' ,'APTT-pre' ,'FBG-pre' ,'INR-pre' ,'PA-post' ,'PT-post' ,'PR-post' ,'APTT-post' ,'FBG-post' ,'INR-post' ,'D-Dimer-post' ,'PA-1' ,'PT-1' ,'PR-1' ,'APTT-1' ,'FBG-1' ,'INR-1' ,'D-Dimer-1' ,'PA-2' ,'PT-2' ,'PR-2' ,'APTT-2' ,'FBG-2' ,'INR-2' ,'D-Dimer-2' ,'PA-3' ,'PT-3' ,'PR-3' ,'APTT-3' ,'FBG-3' ,'INR-3' ,'D-Dimer-3' ,'PA-4' ,'PT-4' ,'PR-4' ,'APTT-4' ,'FBG-4' ,'INR-4' ,'D-Dimer-4' ,'PA-5' ,'PT-5' ,'PR-5' ,'APTT-5' ,'FBG-5' ,'INR-5' ,'D-Dimer-5' ,'PA-6' ,'PT-6' ,'PR-6' ,'APTT-6' ,'FBG-6' ,'INR-6' ,'PA-7' ,'PT-7' ,'PR-7' ,'APTT-7' ,'FBG-7' ,'INR-7',
    # 术后输血情况
    '红细胞POD0' ,'红细胞POD1' ,'红细胞POD2' ,'红细胞POD3' ,'红细胞POD4' ,'红细胞POD5' ,'红细胞POD6' ,'红细胞POD7' ,'红细胞POD8' ,'红细胞POD9' ,'红细胞POD10' ,'红细胞POD11' ,'红细胞POD12' ,'红细胞POD13' ,'红细胞POD14' ,'红细胞POD14+' ,'术后红细胞总量' ,'血浆POD0' ,'血浆POD1' ,'血浆POD2' ,'血浆POD3' ,'血浆POD4' ,'血浆POD5' ,'血浆POD6' ,'血浆POD7' ,'血浆POD8' ,'血浆POD9' ,'血浆POD10' ,'血浆POD11' ,'血浆POD12' ,'血浆POD13' ,'血浆POD14' ,'血浆POD14+' ,'术后血浆总量' ,'血小板POD0' ,'血小板POD1' ,'血小板POD2' ,'血小板POD3' ,'血小板POD4' ,'血小板POD5' ,'血小板POD6' ,'血小板POD7' ,'血小板POD8' ,'血小板POD9' ,'血小板POD10' ,'血小板POD11' ,'血小板POD13' ,'血小板POD14' ,'血小板POD14+']

important_column_en = ['Age', 'Height', 'Weight', 'BMI', 'Type O blood', 'Type A blood' ,'Type O blood', 'Type AB blood' ,'Hepatitis B carrier' ,
# 术中情况
'Surgical approach (Classic 1 Piggyback 0)', 'Operation time min' ,'Anhepatic period min' ,'Warm ischemia time min' ,'Cold ischemia time min', 'Red blood cells', 'Plasma', 'Autologous blood',
 '4% Albumin' ,'2% Albumin', 'Albumin g', 'NS' ,'LR', 'Voluven' ,'Gelofusine', 'Amount of input', 'Amount of bleeding', 'Pleural effusion' ,'Ascites', 'Total urine output',
'Furosemide mg', 'Mannitol ml', 'Sodium bicarbonate ml', 'Fibrinogen g' ,'Prothrombin complex U',
 'Factor VII' ,'Tranexamic acid g/h', 'Tranexamic acid into the pot g' ,'Norepinephrine maintenance' ,'Norepinephrine out of room', 'Adrenaline maintenance', 'Adrenaline out of the room',
 'Dopamine maintenance mg/h' ,'Dopamine out of the room' ,'Atropine when open' ,'Minimum heart rate when open' ,'Minimum SBP when open' ,'Minimum DBP when open', 'Minimum MBP when open',
 'Post-reperfusion syndrome', 'Spleen cut', 'Combined liver and kidney transplantation' ,'Terlipressin ml/h',
# 血常规
'Hb-pre' ,'HCT-pre' ,'MCV-pre' ,'MCH-pre' ,'MCHC-pre' ,'RDW-CVO-pre' ,'PLT-pre' ,'MPV-pre' ,'PDW-pre' ,'LCR-pre' ,'Hb-post' ,'HCT-post' ,'MCV-post' ,'MCH-post' ,'MCHC-post' ,'RDW-CVO-post' ,'PLT-post' ,'MPV-post' ,'PDW-post' ,'LCR-post' ,
'Hb-1' ,'HCT-1' ,'MCV-1' ,'MCH-1' ,'MCHC-1' ,'RDW-CVO-1' ,'PLT-1' ,'MPV-1' ,'PDW-1' ,'LCR-1' ,'Hb-2' ,'HCT-2' ,'MCV-2' ,'MCH-2' ,'MCHC-2' ,'RDW-CVO-2' ,'PLT-2' ,'MPV-2' ,'PDW-2' ,'LCR-2' ,'Hb-3' ,'HCT-3' ,'MCV-3' ,'MCH-3' ,'MCHC-3' ,'RDW-CVO-3' ,'PLT-3' ,'MPV-3' ,'PDW-3' ,'LCR-3' ,'Hb-4' ,'HCT-4' ,'MCV-4' ,'MCH-4' ,'MCHC-4' ,'RDW-CVO-4' ,'PLT-4' ,'MPV-4' ,'PDW-4' ,'LCR-4' ,'Hb-5' ,'HCT-5' ,'MCV-5' ,'MCH-5' ,'MCHC-5' ,'RDW-CVO-5' ,'PLT-5' ,'MPV-5' ,'PDW-5' ,'LCR-5' ,'Hb-6' ,'HCT-6' ,'MCV-6' ,'MCH-6' ,'MCHC-6' ,'RDW-CVO-6' ,'PLT-6' ,'MPV-6' ,'PDW-6' ,'LCR-6' ,'Hb-7' ,'HCT-7' ,'MCV-7' ,'MCH-7' ,'MCHC-7' ,'RDW-CVO-7' ,'PLT-7' ,'MPV-7' ,'PDW-7' ,'LCR-7' ,'Hb-14' ,'HCT-14' ,'MCV-14' ,'MCH-14' ,'MCHC-14' ,'RDW-CVO-14' ,'PLT-14' ,'MPV-14' ,'PDW-14' ,'LCR-14',

#  生化
'AST-pre' ,'ALT-pre' ,'TBIL-pre' ,'ALB-pre' ,'BUN-pre' ,'Cr-pre' ,'Glu-pre' ,'K-pre' ,'Na-pre' ,'Ca-pre' ,'AST-post' ,'ALT-post' ,'TBIL-post' ,'ALB-post' ,'BUN-post' ,'Cr-post' ,'K-post' ,'Na-post' ,'Ca-post' ,'AST-1' ,'ALT-1' ,'TBIL-1' ,'ALB-1' ,'BUN-1' ,'Cr-1' ,'K-1' ,'Na-1' ,'Ca-1' ,'AST-2' ,'ALT-2' ,'TBIL-2' ,'ALB-2' ,'BUN-2' ,'Cr-2' ,'K-2' ,'Na-2' ,'Ca-2' ,'AST-3' ,'ALT-3' ,'TBIL-3' ,'ALB-3' ,'BUN-3' ,'Cr-3' ,'K-3' ,'Na-3' ,'Ca-3' ,'AST-4' ,'ALT-4' ,'TBIL-4' ,'ALB-4' ,'BUN-4' ,'Cr-4' ,'K-4' ,'Na-4' ,'Ca-4' ,'AST-5' ,'ALT-5' ,'TBIL-5' ,'ALB-5' ,'BUN-5' ,'Cr-5' ,'K-5' ,'Na-5' ,'Ca-5' ,'AST-6' ,'ALT-6' ,'TBIL-6' ,'ALB-6' ,'BUN-6' ,'Cr-6' ,'K-6' ,'Na-6' ,'Ca-6' ,'AST-7' ,'ALT-7' ,'TBIL-7' ,'ALB-7' ,'BUN-7' ,'Cr-7' ,'K-7' ,'Na-7' ,'Ca-7' ,'AST-14' ,'ALT-14' ,'TBIL-14' ,'ALB-14' ,'BUN-14' ,'Cr-14' ,'K-14' ,'Na-14' ,'Ca-14',

# 血气
'HCT-pre','PH-pre' ,'PCO2-pre' ,'PO2-pre' ,'Na-pre.1' ,'K-pre.1' ,'Ca-pre.1' ,'Glu-pre.1' ,'Lac-pre' ,'BE(B)-pre' ,'Hb-pre.1' ,'PH-0' ,'PCO2-0' ,'PO2-0' ,'Na-0' ,'K-0' ,'Ca-0' ,'Glu-0' ,'Lac-0' ,'Hct-0' ,'BE(B)-0' ,'Hb-0' ,'PH-30' ,'PCO2-30' ,'PO2-30' ,'Na-30' ,'K-30' ,'Ca-30' ,'Glu-30' ,'Lac-30' ,'Hct-30' ,'BE(B)-30' ,'Hb-30' ,'PH-60' ,'PCO2-60' ,'PO2-60' ,'Na-60' ,'K-60' ,'Ca-60' ,'Glu-60' ,'Lac-60' ,'Hct-60' ,'BE(B)-60' ,'Hb-60' ,'PH-150' ,'PCO2-150' ,'PO2-150' ,'Na-150' ,'K-150' ,'Ca-150' ,'Glu-150' ,'Lac-150' ,'PH-end' ,'PCO2-end' ,'PO2-end' ,'Na-end' ,'K-end' ,'Ca-end' ,'Glu-end' ,'Lac-end' ,'Hct-end' ,'BE(B)-end' ,'Hb-end' ,'PH-icu' ,'PCO2-icu' ,'PO2-icu' ,'Na-icu' ,'K-icu' ,'Ca-icu' ,'Glu-icu' ,'Lac-icu' ,'Hct-icu' ,'BE(B)-icu' ,'Hb-icu' ,'PH-1d' ,'PCO2-1d' ,'PO2-1d' ,'Na-1d' ,'K-1d' ,'Ca-1d' ,'Glu-1d' ,'Lac-1d' ,'Hct-1d' ,'BE(B)-1d' ,'Hb-1d' ,'PH-2d' ,'PCO2-2d' ,'PO2-2d' ,'Na-2d' ,'K-2d' ,'Ca-2d' ,'Glu-2d' ,'Lac-2d' ,'Hct-2d' ,'BE(B)-2d' ,'Hb-2d',

# 凝血
'PA-pre' ,'PT-pre' ,'PR-pre' ,'APTT-pre' ,'FBG-pre' ,'INR-pre' ,'PA-post' ,'PT-post' ,'PR-post' ,'APTT-post' ,'FBG-post' ,'INR-post' ,'D-Dimer-post' ,'PA-1' ,'PT-1' ,'PR-1' ,'APTT-1' ,'FBG-1' ,'INR-1' ,'D-Dimer-1' ,'PA-2' ,'PT-2' ,'PR-2' ,'APTT-2' ,'FBG-2' ,'INR-2' ,'D-Dimer-2' ,'PA-3' ,'PT-3' ,'PR-3' ,'APTT-3' ,'FBG-3' ,'INR-3' ,'D-Dimer-3' ,'PA-4' ,'PT-4' ,'PR-4' ,'APTT-4' ,'FBG-4' ,'INR-4' ,'D-Dimer-4' ,'PA-5' ,'PT-5' ,'PR-5' ,'APTT-5' ,'FBG-5' ,'INR-5' ,'D-Dimer-5' ,'PA-6' ,'PT-6' ,'PR-6' ,'APTT-6' ,'FBG-6' ,'INR-6' ,'PA-7' ,'PT-7' ,'PR-7' ,'APTT-7' ,'FBG-7' ,'INR-7',
# 术后输血情况
'Red blood cells POD0', 'Red blood cells POD1', 'Red blood cells POD2' , 'Red blood cells POD3' , 'Red blood cells POD4' , 'Red blood cells POD5' , 'Red blood cells POD6', 'Red blood cells POD7', 'Red blood cells POD8' , 'Red blood cells POD9' , 'Red blood cells POD10', 'Red blood cells POD11', 'Red blood cells POD12', 'Red blood cells POD13',
'Red blood cells POD14', 'Red blood cells POD14+', 'Postoperative Red blood cells  (Total)',
'Plasma POD0' , 'Plasma POD1' , 'Plasma POD2' , 'Plasma POD3' , 'Plasma POD4', 'Plasma POD5', 'Plasma POD6' , 'Plasma POD7' , 'Plasma POD8', 'Plasma POD9', 'Plasma POD10', 'Plasma POD11', 'Plasma POD12' , 'Plasma POD13' , 'Plasma POD14' , 'Plasma POD15' , 'Postoperative Plasma  (Total)',
'Platelets POD0' ,'Platelets POD1' ,'Platelets POD2' ,'Platelets POD3' ,'Platelets POD4' ,'Platelets POD5' ,'Platelets POD6' ,'Platelets POD7' ,'Platelets POD8' ,'Platelets POD9' ,'Platelets POD10' ,'Platelets POD11' ,'Platelets POD13' ,'Platelets POD14' ,'Platelets POD14+']

TCA_column=['年龄','身高', '体重', 'BMI', 'O型', 'A型' ,'B型', 'AB型' ,'乙肝携带' ,
'术式(经典1背驮0)', '手术时间min' ,'无肝期时间min' ,'热缺血时间min' ,'冷缺血时间min', '红细胞', '血浆', '自体血',
 '4%白蛋白' ,'2%白蛋白', '纯白蛋白g', 'NS' ,'LR', '万汶' ,'佳乐施', '总入量', '出血量', '胸水' ,'腹水', '总尿量',
'速尿mg', '甘露醇ml', '碳酸氢钠ml', '纤维蛋白原g' ,'凝血酶原复合物U',
 'VII因子' ,'氨甲环酸g/h', '氨甲环酸入壶g' ,'去甲肾上腺素维持' ,'去甲肾上腺素出室', '肾上腺素维持', '肾上腺素出室',
 '多巴胺维持mg/h' ,'多巴胺出室' ,'开放时阿托品' ,'开放时最低心率' ,'开放时最低SBP' ,'开放时最低DBP', '开放时最低MBP',
 '再灌注后综合征', '切脾', '肝肾联合移植' ,'特利加压素ml/h',

# 血常规
'Hb-pre' ,'HCT-pre', 'MCV-pre', 'MCH-pre' ,'MCHC-pre' ,'RDW-CVO-pre', 'PLT-pre', 'MPV-pre' ,'PDW-pre','LCR-pre', 
 'Hb-post', 'HCT-post', 'MCV-post' ,'MCH-post', 'MCHC-post',
 'RDW-CVO-post', 'PLT-post' ,'MPV-post' ,'PDW-post', 'LCR-post',
 'Hb-1,2', 'HCT-1,2' ,'MCV-1,2' ,'MCH-1,2', 'MCHC-1,2',
 'RDW-CVO-1,2', 'PLT-1,2', 'MPV-1,2','PDW-1,2' ,'LCR-1,2','Hb-3,4' ,'HCT-3,4', 'MCV-3,4', 'MCH-3,4', 'MCHC-3,4',
 'RDW-CVO-3,4', 'PLT-3,4' ,'MPV-3,4', 'PDW-3,4' ,'LCR-3,4','Hb-5,6',
 'HCT-5,6', 'MCV-5,6', 'MCH-5,6' ,'MCHC-5,6', 'RDW-CVO-5,6', 'PLT-5,6', 'MPV-5,6' ,'PDW-5,6',
 'LCR-5,6','Hb-7,14', 'HCT-7,14' ,'MCV-7,14' ,'MCH-7,14' ,'MCHC-7,14','RDW-CVO-7,14', 'PLT-7,14' ,'MPV-7,14', 'PDW-7,14' ,'LCR-7,14',
#  生化
#  'AST-pre', 'ALT-pre','TBIL-pre' ,'ALB-pre', 'BUN-pre', 'Cr-pre' ,'Glu-pre_x' ,'K-pre_x' ,'Na-pre_x','Ca-pre_x',
 'AST-pre', 'ALT-pre','TBIL-pre' ,'ALB-pre', 'BUN-pre', 'Cr-pre' ,'Glu-pre' ,'K-pre' ,'Na-pre','Ca-pre',
 'AST-post', 'ALT-post' ,'TBIL-post', 'ALB-post' ,'BUN-post','Cr-post', 'K-post', 'Na-post' ,'Ca-post',
 'AST-1,2' ,'ALT-1,2' ,'TBIL-1,2' ,'ALB-1,2','BUN-1,2', 'Cr-1,2' ,'K-1,2' ,'Na-1,2' ,'Ca-1,2',
 'AST-3,4' ,'ALT-3,4', 'TBIL-3,4' ,'ALB-3,4','BUN-3,4', 'Cr-3,4', 'K-3,4' ,'Na-3,4', 'Ca-3,4',
'AST-5,6', 'ALT-5,6', 'TBIL-5,6', 'ALB-5,6','BUN-5,6','Cr-5,6', 'K-5,6' ,'Na-5,6', 'Ca-5,6',
'AST-7,14' ,'ALT-7,14' ,'TBIL-7,14' ,'ALB-7,14','BUN-7,14' ,'Cr-7,14', 'K-7,14', 'Na-7,14' ,'Ca-7,14',
# 血气
'PH-pre','PCO2-pre','PO2-pre','Na-pre.1','K-pre.1','Ca-pre.1','Glu-pre.1','Lac-pre','Hct-pre','BE(B)-pre', 'Hb-pre.1',
'PH-0','PCO2-0','PO2-0','Na-0','K-0','Ca-0','Glu-0','Lac-0','Hct-0','BE(B)-0','Hb-0',

'PH-30,60','PCO2-30,60','PO2-30,60','Na-30,60','K-30,60','Ca-30,60','Glu-30,60','Lac-30,60','Hct-30,60','BE(B)-30,60','Hb-30,60',

'PH-end,icu','PCO2-end,icu','PO2-end,icu','Na-end,icu','K-end,icu','Ca-end,icu','Glu-end,icu','Lac-end,icu','Hct-end,icu','BE(B)-end,icu','Hb-end,icu',
'PH-1d,2d' ,'PCO2-1d,2d' ,'PO2-1d,2d' ,'Na-1d,2d' ,'K-1d,2d', 'Ca-1d,2d' ,'Glu-1d,2d','Lac-1d,2d', 'Hct-1d,2d' ,'BE(B)-1d,2d', 'Hb-1d,2d',

'PA-pre','PT-pre','PR-pre','APTT-pre','FBG-pre','INR-pre' ,
 'PA-post','PT-post','PR-post','APTT-post','FBG-post','INR-post','D-Dimer-post',

'PA-1,2','PT-1,2','PR-1,2','APTT-1,2','FBG-1,2','INR-1,2','D-Dimer-1,2',
'PA-3,4' ,'PT-3,4', 'PR-3,4', 'APTT-3,4', 'FBG-3,4' ,'INR-3,4', 'D-Dimer-3,4',
'PA-6,7', 'PT-6,7' ,'PR-6,7' ,'APTT-6,7','FBG-6,7', 'INR-6,7',

'红细胞POD0,1', '红细胞POD2,3' , '红细胞POD4,5' , '红细胞POD6,7','红细胞POD8,9' , '红细胞POD10,11', '红细胞POD12,13',
'红细胞POD14,14+', '术后红细胞总量',
'血浆POD0,1' , '血浆POD2,3' ,'血浆POD4,5', '血浆POD6,7' , '血浆POD8,9','血浆POD10,11','血浆POD12,13' , '血浆POD14,15' , '术后血浆总量',
'血小板POD0,1' ,'血小板POD2,3' , '血小板POD4,5' , '血小板POD6,7' , '血小板POD8,9','血小板POD10,11' , '血小板POD13,14','血小板POD14+'
]
TCA_column_en=['Age', 'Height', 'Weight', 'BMI', 'Type O blood', 'Type A blood' ,'Type B blood', 'Type AB blood' ,'Hepatitis B carrier' ,
'Surgical approach (Classic 1 Piggyback 0)', 'Operation time min' ,'Anhepatic period min' ,'Warm ischemia time min' ,'Cold ischemia time min', 'Red blood cells', 'Plasma', 'Autologous blood',
 '4% Albumin' ,'2% Albumin', 'Albumin g', 'NS' ,'LR', 'Voluven' ,'Gelofusine', 'Amount of input', 'Amount of bleeding', 'Pleural effusion' ,'Ascites', 'Total urine output',
'Furosemide mg', 'Mannitol ml', 'Sodium bicarbonate ml', 'Fibrinogen g' ,'Prothrombin complex U',
 'Factor VII' ,'Tranexamic acid g/h', 'Tranexamic acid into the pot g' ,'Norepinephrine maintenance' ,'Norepinephrine out of room', 'Adrenaline maintenance', 'Adrenaline out of the room',
 'Dopamine maintenance mg/h' ,'Dopamine out of the room' ,'Atropine when open' ,'Minimum heart rate when open' ,'Minimum SBP when open' ,'Minimum DBP when open', 'Minimum MBP when open',
 'Post-reperfusion syndrome', 'Spleen cut', 'Combined liver and kidney transplantation' ,'Terlipressin ml/h',

# 血常规
'Hb-pre' ,'HCT-pre', 'MCV-pre', 'MCH-pre' ,'MCHC-pre' ,'RDW-CVO-pre', 'PLT-pre', 'MPV-pre' ,'PDW-pre','LCR-pre', 
 'Hb-post', 'HCT-post', 'MCV-post' ,'MCH-post', 'MCHC-post',
 'RDW-CVO-post', 'PLT-post' ,'MPV-post' ,'PDW-post', 'LCR-post',
 'Hb-1,2', 'HCT-1,2' ,'MCV-1,2' ,'MCH-1,2', 'MCHC-1,2',
 'RDW-CVO-1,2', 'PLT-1,2', 'MPV-1,2','PDW-1,2' ,'LCR-1,2','Hb-3,4' ,'HCT-3,4', 'MCV-3,4', 'MCH-3,4', 'MCHC-3,4',
 'RDW-CVO-3,4', 'PLT-3,4' ,'MPV-3,4', 'PDW-3,4' ,'LCR-3,4','Hb-5,6',
 'HCT-5,6', 'MCV-5,6', 'MCH-5,6' ,'MCHC-5,6', 'RDW-CVO-5,6', 'PLT-5,6', 'MPV-5,6' ,'PDW-5,6',
 'LCR-5,6','Hb-7,14', 'HCT-7,14' ,'MCV-7,14' ,'MCH-7,14' ,'MCHC-7,14','RDW-CVO-7,14', 'PLT-7,14' ,'MPV-7,14', 'PDW-7,14' ,'LCR-7,14',
#  生化
#  'AST-pre', 'ALT-pre','TBIL-pre' ,'ALB-pre', 'BUN-pre', 'Cr-pre' ,'Glu-pre_x' ,'K-pre_x' ,'Na-pre_x','Ca-pre_x',
 'AST-pre', 'ALT-pre','TBIL-pre' ,'ALB-pre', 'BUN-pre', 'Cr-pre' ,'Glu-pre' ,'K-pre' ,'Na-pre','Ca-pre',
 'AST-post', 'ALT-post' ,'TBIL-post', 'ALB-post' ,'BUN-post','Cr-post', 'K-post', 'Na-post' ,'Ca-post',
 'AST-1,2' ,'ALT-1,2' ,'TBIL-1,2' ,'ALB-1,2','BUN-1,2', 'Cr-1,2' ,'K-1,2' ,'Na-1,2' ,'Ca-1,2',
 'AST-3,4' ,'ALT-3,4', 'TBIL-3,4' ,'ALB-3,4','BUN-3,4', 'Cr-3,4', 'K-3,4' ,'Na-3,4', 'Ca-3,4',
'AST-5,6', 'ALT-5,6', 'TBIL-5,6', 'ALB-5,6','BUN-5,6','Cr-5,6', 'K-5,6' ,'Na-5,6', 'Ca-5,6',
'AST-7,14' ,'ALT-7,14' ,'TBIL-7,14' ,'ALB-7,14','BUN-7,14' ,'Cr-7,14', 'K-7,14', 'Na-7,14' ,'Ca-7,14',
# 血气
'PH-pre','PCO2-pre','PO2-pre','Na-pre.1','K-pre.1','Ca-pre.1','Glu-pre.1','Lac-pre','Hct-pre','BE(B)-pre', 'Hb-pre.1',
'PH-0','PCO2-0','PO2-0','Na-0','K-0','Ca-0','Glu-0','Lac-0','Hct-0','BE(B)-0','Hb-0',

'PH-30,60','PCO2-30,60','PO2-30,60','Na-30,60','K-30,60','Ca-30,60','Glu-30,60','Lac-30,60','Hct-30,60','BE(B)-30,60','Hb-30,60',

'PH-end,icu','PCO2-end,icu','PO2-end,icu','Na-end,icu','K-end,icu','Ca-end,icu','Glu-end,icu','Lac-end,icu','Hct-end,icu','BE(B)-end,icu','Hb-end,icu',
'PH-1d,2d' ,'PCO2-1d,2d' ,'PO2-1d,2d' ,'Na-1d,2d' ,'K-1d,2d', 'Ca-1d,2d' ,'Glu-1d,2d','Lac-1d,2d', 'Hct-1d,2d' ,'BE(B)-1d,2d', 'Hb-1d,2d',

'PA-pre','PT-pre','PR-pre','APTT-pre','FBG-pre','INR-pre' ,
 'PA-post','PT-post','PR-post','APTT-post','FBG-post','INR-post','D-Dimer-post',

'PA-1,2','PT-1,2','PR-1,2','APTT-1,2','FBG-1,2','INR-1,2','D-Dimer-1,2',
'PA-3,4' ,'PT-3,4', 'PR-3,4', 'APTT-3,4', 'FBG-3,4' ,'INR-3,4', 'D-Dimer-3,4',
'PA-6,7', 'PT-6,7' ,'PR-6,7' ,'APTT-6,7','FBG-6,7', 'INR-6,7',

'Red blood cells POD0,1', 'Red blood cells POD2,3' , 'Red blood cells POD4,5' , 'Red blood cells POD6,7','Red blood cells POD8,9' , 'Red blood cells POD10,11', 'Red blood cells POD12,13',
'Red blood cells POD14,14+', 'Postoperative Red blood cells  (Total)',
'Plasma POD0,1' , 'Plasma POD2,3' ,'Plasma POD4,5', 'Plasma POD6,7' , 'Plasma POD8,9','Plasma POD10,11','Plasma POD12,13' , 'Plasma POD14,15' , 'Postoperative Plasma  (Total)',
'Platelets POD0,1' ,'Platelets POD2,3' , 'Platelets POD4,5' , 'Platelets POD6,7' , 'Platelets POD8,9','Platelets POD10,11' , 'Platelets POD13,14','Platelets POD14+'
]

# np.savetxt('./data/wrapper_experiment/important_features_name.txt', important_column, delimiter=",",fmt='%s')
np.savetxt('./data/wrapper_experiment/important_features_name.txt', important_column,fmt='%s')
# np.savetxt('./data/wrapper_experiment/TCA+SVM_features_name.txt', TCA_column,fmt='%s')
np.savetxt('./data/wrapper_experiment/TCA+SVM_features_name_en.txt', TCA_column_en,fmt='%s')
important_column_en = pd.DataFrame(important_column_en)
important_column_en.to_csv('./data/output/important_features_name_en.csv', index=False)   
# important_column = pd.DataFrame(important_column)
# important_column.to_csv('./data/output/important_features_name_cn.csv', index=False, encoding='utf_8_sig')   


# # 读取测试一下保存的不会有问题
# # read_important_column=np.loadtxt('./data/wrapper_experiment/important_features_name.txt', delimiter=',',dtype='str')
# read_important_column=np.loadtxt('./data/wrapper_experiment/important_features_name.txt',dtype='str')
# TCA_column_en=np.loadtxt('./data/wrapper_experiment/TCA+SVM_features_name_en.txt',dtype='str', delimiter='\n')

# print(TCA_column_en)
# print(TCA_column_en.shape)

data = pd.read_csv('./data/fill_NA_features.csv')
PCA_data = data.loc[:,important_column]
PCA_data.to_csv('./data/data_needed_features.csv', index=False)