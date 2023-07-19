# -*- coding: utf-8 -*-
import math
import time
import json
import copy
# import warnings
import datetime
import gurobipy as gp
from gurobipy import *
import pandas as pd
from collections import defaultdict

'''

数据读取与相关参数处理

'''
contractorinfoall = pd.read_json('in/ContractorInfo.json')
appliedcaseinfoall = pd.read_json('in/AppliedCaseInfo.json')
configinfo = pd.read_json('in/ConfigInfo.json')
currentcaseinfoall = pd.read_json('in/CurrentCaseInfo.json')

# 参数读取与设置
lowstart = configinfo.loc[0, 'LowStart']
low_start = datetime.datetime.strptime(lowstart, '%Y/%m/%d')
lowend = configinfo.loc[0, 'LowEnd']
low_end = datetime.datetime.strptime(lowend, '%Y/%m/%d')
SameTimeLength = configinfo.loc[0, 'SameTimeLength']
propdeviation = eval(configinfo.loc[0, 'PropDeviaion'])
quantideviation = eval(configinfo.loc[0, 'QuantiDeviation'])
SameCityDay = int(configinfo.loc[0, 'SameCityDay'])
newyearstartday = datetime.datetime.strptime(configinfo.loc[0, 'NewYearStartDay'], '%Y/%m/%d')
ContractStartDay = datetime.datetime.strptime(configinfo.loc[0, 'ContractStartDay'], '%Y/%m/%d')
nosamecity = configinfo.loc[0, 'NotApplicableCity'][1:-1].split(',')

# 标段集合
contractorinfoall.SectionCode = contractorinfoall['SectionCode'].apply(lambda x: '{:07d}'.format(x))
appliedcaseinfoall.SectionCode = appliedcaseinfoall['SectionCode'].apply(lambda x: '{:07d}'.format(x))
currentcaseinfoall.SectionCode = currentcaseinfoall['SectionCode'].apply(lambda x: '{:07d}'.format(x))
section_list = list(set(contractorinfoall['SectionCode']))
# print(type(section_list[0]))
# 相关时间处理
allstart = time.time()
today = datetime.date.today()
nowday = datetime.datetime.strptime(str(today), '%Y-%m-%d')

# 新年约束起止时间in 6
newyear_cntr_starttime = newyearstartday - datetime.timedelta(days=30)
newyear_cntr_endtime = newyearstartday + datetime.timedelta(days=20)

# 低峰月约束起止时间(派案日期在[低峰月开始日期-start_low(day),低峰月结束日期-end——low(day)]之间才考虑是否需要满足低峰月约束)
start_low = 30
end_low = 10

# 优先级设定
priority_dic = defaultdict(int)
priority_dic['obj_new_year'] = 11
priority_dic['obj_new_start'] = 10
priority_dic['obj_case_num'] = 9
priority_dic['obj_interval_gap'] = 8
priority_dic['obj_reset'] = 7
priority_dic['obj_same_address'] = 6
priority_dic['obj_min_distance'] = 5
priority_dic['obj_sum_share_gap'] = 4
priority_dic['obj_max_gap'] = 3
priority_dic['obj_same_city'] = 2
priority_dic['obj_low_month'] = 1

# gap设定
gap_dic = defaultdict(float)
gap_dic['obj_new_year'] = 0
gap_dic['obj_new_start'] = 0
gap_dic['obj_case_num'] = 0
gap_dic['obj_interval_gap'] = 0
gap_dic['obj_reset'] = 0
gap_dic['obj_same_address'] = 0
gap_dic['obj_min_distance'] = 0
gap_dic['obj_sum_share_gap'] = 0.0001
gap_dic['obj_max_gap'] = 0.0001
gap_dic['obj_same_city'] = 0
gap_dic['obj_low_month'] = 0

# 约束设定(new_start_cntr和low_cntr会根据派案时间更新)
maxnum_cntr = True
new_start_cntr = True
new_year_cntr = False
share_interval_gap_cntr = True
reset_cntr = True
address_cntr = True
share_max_gap_cntr = True
share_sum_gap_cntr = True
high_maxnum_cntr = True
city_cntr = True
average_num_cntr = True
average_area_cntr = True
low_cntr = False

# out
path1 = 'out'
path2 = 'log'
if not os.path.exists(path1):
    os.makedirs(path1)
if not os.path.exists(path2):
    os.makedirs(path2)
jsresult = []

'''

新旧合同期数据划分

'''

contractor_split = {}
appliedcase_split = {}
# "ExpirationDate'为自己增加的字段，为了区分新旧合同期
contractorinfoall.insert(loc=contractorinfoall.shape[1], column="ExpirationDate", value='2024/4/27')
appliedcaseinfoall['ApproachDateUpdate'] = appliedcaseinfoall['ApproachDate'].apply(
    lambda x: datetime.datetime.strptime(x[:10], '%Y/%m/%d'))
contractorinfoall['ExpirationDate'] = contractorinfoall['ExpirationDate'].apply(
    lambda x: datetime.datetime.strptime(x[:10], '%Y/%m/%d'))
contractor_split['old'] = contractorinfoall[contractorinfoall['ExpirationDate'] <= ContractStartDay]
appliedcase_split['old'] = appliedcaseinfoall[appliedcaseinfoall['ApproachDateUpdate'] < ContractStartDay]
flag = 'cur'
if contractor_split['old'].empty:
    if not appliedcase_split['old'].empty:
        appliedcase_notallot = appliedcase_split['old']
        casecodelist_0 = list(appliedcase_notallot['CaseCode'])
        for j in casecodelist_0:
            theline = {}
            theline['CaseCode'] = j
            theline['ContractorCode'] = ''
            theline['AllotLabel'] = '1'
            theline['AllotReason'] = 'NoAllotContractor'
            jsresult.append(theline)
else:
    flag = 'new'
contractor_split[flag] = contractorinfoall[contractorinfoall['ExpirationDate'] > ContractStartDay]
appliedcase_split[flag] = appliedcaseinfoall[appliedcaseinfoall['ApproachDateUpdate'] >= ContractStartDay]


def contractor_data_process(contractorinfoall, contractorcode_name, section_contractorcode_name, contractor_property,
                            current_case_num, axis, maxsamenum, target_share, current_share, current_num,
                            section_total_num_h, section_list, ):
    # warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)
    contractorinfo = contractorinfoall
    contractorinfo.loc[:, 'ContractorCodeKey'] = contractorinfoall.apply(
        lambda x: str(x['ContractorCode']) + str(x['SectionCode']), axis=1)
    contractorinfo.loc[:, 'ContractorNameKey'] = contractorinfo.apply(
        lambda x: str(x['ContractorName']) + str(x['SectionCode']),
        axis=1)
    contractorinfo.ContractorCodeKey = (contractorinfo['ContractorCodeKey']).astype(str)
    contractorinfo.ContractorNameKey = (contractorinfo['ContractorNameKey']).astype(str)
    contractorinfo.SectionCode = (contractorinfo['SectionCode']).astype(str)
    temp_contractor = contractorinfo[['ContractorCodeKey', 'ContractorNameKey']].drop_duplicates()
    contractorcode_name.update(temp_contractor.set_index('ContractorCodeKey')['ContractorNameKey'].to_dict())

    # 按标段对厂商进行划分
    for r in section_list:
        tempinfo = contractorinfo[contractorinfo['SectionCode'] == r]
        section_contractor = tempinfo[['ContractorCodeKey', 'ContractorNameKey']].drop_duplicates()
        section_contractorcode_name[r] = section_contractor.set_index('ContractorCodeKey')[
            'ContractorNameKey'].to_dict()
        # print(section_contractorcode_name[r])
    # 厂商信息读取

    # 已服务的各类型店的数量信息
    for i in contractorcode_name.keys():
        contractor_property[i]['NewNum'] = contractorinfo[contractorinfo['ContractorCodeKey'] == i].NewCaseNum.values[0]
        contractor_property[i]['MIENum'] = contractorinfo[contractorinfo['ContractorCodeKey'] == i].MIENum.values[0]
        contractor_property[i]['PIENum'] = contractorinfo[contractorinfo['ContractorCodeKey'] == i].PIENum.values[0]
        # 厂商同期最大施工数
        maxsamenum[i] = contractorinfo[contractorinfo['ContractorCodeKey'] == i].MaxSametimeCaseNum.values[0]
        # 份额信息
        ts = contractorinfo[contractorinfo['ContractorCodeKey'] == i].TargetShare.values[0]
        target_share[i] = float(ts)
        current_share[i] = contractorinfo[contractorinfo['ContractorCodeKey'] == i].CurrentShare.values[0]
        current_num[i] = contractorinfo[contractorinfo['ContractorCodeKey'] == i].CurrentCaseNum.values[0]
        id = contractorinfo[contractorinfo['ContractorCodeKey'] == i].index[0]
        current_case_num[i] = contractorinfo.loc[id, 'NewCaseNum'] + contractorinfo.loc[id, 'MIENum'] + \
                              contractorinfo.loc[id, 'PIENum']
        id = contractorinfo[contractorinfo['ContractorCodeKey'] == i].index[0]
        k = contractorinfoall.loc[id, 'Coordinate'][1:-1].replace('{', '').replace('}', '').split(',')
        k = [float(i) for i in k]
        axis[i] = k

    # 标段案子数（历史案）
    for r in section_list:
        section_total_num_h[r] = contractorinfo[contractorinfo['SectionCode'] == r].CurrentCaseNum.sum()
    return contractorinfo


def currentcase_data_process(currentcaseinfo, occupied, contractorcode_name, nowday, SameTimeLength, low_start_Day,
                             low_end_day,
                             cur_newyear_num, newyearstartday, ContractStartDay, cur_newstart_num, cur_low_num,
                             region_casecodelist_h, allot_avail_h, endday_dic, isnew):
    """

    :rtype: object
    """
    currentcaseinfo.dropna(subset=['ContractorCode'], inplace=True)
    currentcaseinfo.ContractorCode = (currentcaseinfo['ContractorCode']).astype(int)
    currentcaseinfo['ContractorCodeKey'] = currentcaseinfo.apply(
        lambda x: str(x['ContractorCode']) + str(x['SectionCode']),
        axis=1)
    currentcaseinfo['ContractorNameKey'] = currentcaseinfo.apply(
        lambda x: str(x['ContractorName']) + str(x['SectionCode']),
        axis=1)
    currentcaseinfo.ContractorCodeKey = (currentcaseinfo['ContractorCodeKey']).astype(str)
    currentcaseinfo.ContractorNameKey = (currentcaseinfo['ContractorNameKey']).astype(str)
    currentcaseinfo.CityCode = (currentcaseinfo['CityCode']).astype(str)

    #  currentcase occupied contractor 第一层遍历是防止currentcase中存在两个案子由同一个厂商负责的情况
    for i in contractorcode_name.keys():
        for id in currentcaseinfo[currentcaseinfo['ContractorCodeKey'] == i].index:
            region_casecodelist_h[currentcaseinfo.loc[id, 'SectionCode']].append(currentcaseinfo.loc[id, 'CaseCode'])
            startdate = currentcaseinfo.loc[id, 'ApproachDate']
            startdate_time = datetime.datetime.strptime(startdate, '%Y/%m/%d')
            enddate = currentcaseinfo.loc[id, 'CompletionDate']
            enddate_time = datetime.datetime.strptime(enddate, '%Y/%m/%d')
            startday = (startdate_time - nowday).days
            if currentcaseinfo.loc[id, 'CaseType'] == 1 or currentcaseinfo.loc[id, 'CaseType'] == 2:
                endday = startday + SameTimeLength
            else:
                endday = (enddate_time - nowday).days + 1
            # print(startday,endday)
            endday_dic[currentcaseinfo.loc[id, 'CaseCode']] = endday
            if (isnew == 'old') | (startdate_time >= ContractStartDay):
                for k in range(max(startday, 0) + 1, max(endday, 0) + 1):
                    occupied[i, k] += 1
            if startdate_time >= newyearstartday:
                cur_newyear_num[i] += 1
            if startdate_time >= ContractStartDay:
                cur_newstart_num[i] += 1
            if low_start_Day <= startdate_time <= low_end_day:
                t = startday + 1
                cur_low_num[i, t] += 1
            allot_avail_h[i, currentcaseinfo.loc[id, 'CaseCode']] = 1

    return currentcaseinfo


def appliedcase_data_process(appliedcaseinfoall, contractorinfo, allot_avail, allot_availset, section_list, starttime,
                             duration, max_day, is_samecity, newyearstartday, is_ctime, region_casecodelist, allot_num,
                             jsresult, nowday, SameTimeLength, nosamecity, is_low, is_new_year, low_start_day,
                             low_end_day, type, distance_dic, axis, case_axis,
                             allot_share_num, isreset, OC, is_sameadress, currentcaseinfo, cur_endday, type_case_dic):
    appliedcaseinfo = appliedcaseinfoall
    appliedcaseinfo.CityCode = (appliedcaseinfo['CityCode']).astype(str)
    appliedcaseinfo.CaseType = (appliedcaseinfo['CaseType']).astype(int)
    appliedcaseinfo.OrigContractor.fillna(value='0', inplace=True)
    # print(appliedcaseinfo.OrigContractor)
    appliedcaseinfo.OrigContractor = (appliedcaseinfo['OrigContractor']).astype(int)
    # appliedcaseinfo.OrigContractor = appliedcaseinfo['OrigContractor'].apply(lambda x:'{:d}'.format(x))
    type_case_dic['NEW_case'] = set(appliedcaseinfo[appliedcaseinfo['CaseType'] == 1].CaseCode)
    type_case_dic['MIE_case'] = set(appliedcaseinfo[appliedcaseinfo['CaseType'] == 2].CaseCode)
    type_case_dic['PIE_case'] = set(appliedcaseinfo[appliedcaseinfo['CaseType'] == 3].CaseCode)
    av = defaultdict(list)
    casecodelist = list(appliedcaseinfo['CaseCode'])

    # appliedcase starttime/duration
    casecodelist_copy = copy.deepcopy(casecodelist)

    for j in casecodelist_copy:
        id = appliedcaseinfo[appliedcaseinfo['CaseCode'] == j].index.values[0]
        case_axis[j] = appliedcaseinfo.loc[id, 'Coordinate'].split(',')
        startdate = appliedcaseinfo[appliedcaseinfo['CaseCode'] == j].ApproachDate.values[0]
        enddate = appliedcaseinfo[appliedcaseinfo['CaseCode'] == j].CompletionDate.values[0]
        if startdate == None or enddate == None:
            theline = {}
            theline['CaseCode'] = j
            theline['ContractorCode'] = ''
            theline['AllotLabel'] = '1'
            theline['AllotReason'] = 'NoDate'
            jsresult.append(theline)
            casecodelist.remove(j)
            appliedcaseinfo = appliedcaseinfo.drop(id, axis=0)
        else:
            startdate_time = datetime.datetime.strptime(startdate, '%Y/%m/%d')
            starttime[j] = (startdate_time - nowday).days + 1
            if starttime[j] < 1:
                theline = {}
                theline['CaseCode'] = j
                theline['ContractorCode'] = ''
                theline['AllotLabel'] = '1'
                theline['AllotReason'] = 'BeforeToday'
                jsresult.append(theline)
                casecodelist.remove(j)
                appliedcaseinfo = appliedcaseinfo.drop(id, axis=0)
            elif starttime[j] > 60:
                theline = {}
                theline['CaseCode'] = j
                theline['ContractorCode'] = ''
                theline['AllotLabel'] = '1'
                theline['AllotReason'] = '>60Day'
                jsresult.append(theline)
                casecodelist.remove(j)
                appliedcaseinfo = appliedcaseinfo.drop(id, axis=0)
            else:
                if startdate_time >= newyearstartday:
                    is_new_year[j] = 1
                if low_start_day <= startdate_time <= low_end_day:
                    is_low[j, starttime[j]] = 1
                enddate = appliedcaseinfo[appliedcaseinfo['CaseCode'] == j].CompletionDate.values[0]
                enddate_time = datetime.datetime.strptime(enddate, '%Y/%m/%d')
                case_type = appliedcaseinfo[appliedcaseinfo['CaseCode'] == j].CaseType.values[0]
                if case_type == 1 or case_type == 2:
                    duration[j] = int(SameTimeLength)
                else:
                    duration[j] = (enddate_time - startdate_time).days + 1
                for t in range(starttime[j], starttime[j] + duration[j]):
                    is_ctime[j, t] = 1
                if max_day[0] < (startdate_time - nowday).days + duration[j]:
                    max_day[0] = (startdate_time - nowday).days + duration[j]
                # print(j,starttime[j],duration[j])
    for id in contractorinfo.index:
        contractorcode = contractorinfo.loc[id, 'ContractorCodeKey']
        sec = contractorinfo.loc[id, 'SectionCode']
        axis_list = axis[contractorcode]
        # marketprov = eval(contractorinfo.loc[id, 'MarketProvCode'])
        avail_case = list(appliedcaseinfo[(appliedcaseinfo['SectionCode'] == str(sec))].CaseCode)
        for case in avail_case:
            distance_dic[contractorcode, case] = 10000000
            allot_avail[contractorcode, case] = 1
            av[case].append(contractorinfo.loc[id, 'SectionCode'])  # 记录此案可被分配至哪个标段
            allot_availset[case].add(contractorcode)  # 记录此案可被分配至哪些厂商
            index = 0
            while index < len(axis_list):
                temp = int(
                    math.sqrt((axis_list[index] - float(case_axis[case][0])) * (
                            axis_list[index] - float(case_axis[case][0])) + (
                                      axis_list[index + 1] - float(case_axis[case][1])) * (
                                      axis_list[index + 1] - float(case_axis[case][1]))))
                if temp < distance_dic[contractorcode, case]:
                    distance_dic[contractorcode, case] = temp
                index += 2

    # 本次派案各市场派案总量
    current_casecode_list = list(currentcaseinfo['CaseCode'])
    casecodelist_copy = copy.deepcopy(casecodelist)
    for j in casecodelist_copy:
        id = appliedcaseinfo[appliedcaseinfo['CaseCode'] == j].index.values[0]
        casecode = appliedcaseinfo.loc[id, 'CaseCode']
        if not allot_availset[casecode]:
            theline = {}
            theline['CaseCode'] = j
            theline['ContractorCode'] = ''
            theline['AllotLabel'] = '1'
            theline['AllotReason'] = 'NoAllotContractor'
            jsresult.append(theline)
            casecodelist.remove(j)
            appliedcaseinfo = appliedcaseinfo.drop(id, axis=0)
            continue

        #  tpye_para的设置，计算份额使用
        if appliedcaseinfo.loc[id, 'CaseType'] == 3:
            type[appliedcaseinfo.loc[id, 'CaseCode']] = 0.5
        else:
            type[appliedcaseinfo.loc[id, 'CaseCode']] = 1

        #  重置案参数设置
        if appliedcaseinfo.loc[id, 'IsReset'] == 1:
            isreset[appliedcaseinfo.loc[id, 'CaseCode']] = 1
            OC[appliedcaseinfo.loc[id, 'CaseCode']] = str(appliedcaseinfo.loc[id, 'OrigContractor'])
            # print(OC[appliedcaseinfo.loc[id, 'CaseCode']])

        #  同地址参数设置
        if not appliedcaseinfo.loc[id, 'SameEstate'] == '':
            estate_id = appliedcaseinfo.loc[id, 'SameEstate']
            if estate_id in current_casecode_list:
                if cur_endday[estate_id] - 1 >= starttime[casecode]:
                    is_sameadress[estate_id, casecode] = 1
                    is_sameadress[casecode, estate_id] = 1
            elif estate_id in casecodelist:
                if ((starttime[estate_id] <= starttime[casecode]) & (
                        starttime[estate_id] + duration[estate_id] - 1 >= starttime[casecode])) | (
                        (starttime[casecode] <= starttime[estate_id]) & (
                        starttime[casecode] + duration[casecode] - 1 >= starttime[estate_id])):
                    is_sameadress[estate_id, casecode] = 1
                    is_sameadress[casecode, estate_id] = 1
        #  同城市参数
        if appliedcaseinfo.loc[id, 'CityCode'] not in nosamecity:
            casetype = appliedcaseinfo.loc[id, 'CaseType']
            for c_id in currentcaseinfo[currentcaseinfo['CityCode'] == appliedcaseinfo.loc[id, 'CityCode']].index:
                cur_casecode = currentcaseinfo.loc[c_id, 'CaseCode']
                if (casetype <= 2 and cur_endday[cur_casecode] - SameTimeLength - 1 <= starttime[casecode] <=
                    cur_endday[cur_casecode] + SameCityDay) or (
                        casetype == 3 and starttime[casecode] <= cur_endday[cur_casecode] + SameCityDay):
                    is_samecity[casecode, cur_casecode] = 2
                    is_samecity[cur_casecode, casecode] = 2
                else:
                    is_samecity[casecode, cur_casecode] = 1
                    is_samecity[cur_casecode, casecode] = 1

    # 此轮申请案子各标段的分布数量
    for r in section_list:
        region_casecodelist[r] = list(appliedcaseinfo[appliedcaseinfo['SectionCode'] == r].CaseCode)
        allot_num[r] = len(appliedcaseinfo[appliedcaseinfo['SectionCode'] == r])
        for id in appliedcaseinfo[appliedcaseinfo['SectionCode'] == r].index:
            if appliedcaseinfo.loc[id, 'CaseType'] == 3:
                allot_share_num[r] += 0.5
            else:
                allot_share_num[r] += 1

    return appliedcaseinfo
    #


def gurobi_processing(I, J, K, T, r, type, maxsamenum, occupied, ICC, ICC_h, IR, OC, ResetProp, ResetQuanti,
                      SE, SC, target_share, priority_dic, jsresult, writer, distance_dic,
                      period_low, cur_low_num, is_start, is_ctime, occupied_0,
                      current_num, is_new_year, cur_newyear_num, key, gap_dic, nowday,
                      area_total_num, share_sum_gap_cntr=True, share_max_gap_cntr=True, share_interval_gap_cntr=True,
                      new_start_cntr=False, new_year_cntr=False, case_num_cntr=True, min_distance_cntr=True,
                      maxnum_cntr=True, reset_cntr=True, address_cntr=True, city_cntr=True,
                      low_cntr=True):
    '''

    I:厂商集合
    J:applied case set
    K:current case set
    T:日期集合
    r:区域
    M:市场集合
    type:若为PIE则为0.5，否则为1
    maxsamenum:最大同期施工数
    occupied:历史施工数记录
    ICC:applied case匹配信息 allot_avail
    ICC_h:current case匹配信息 allot_avail_h
    IR:是否为重置案
    OC[j]:重置案j的原厂商i
    SE:j1 j2 是否满足同地址规则
    SC:j1 j2 是否满足同城市规则
    target_share:目标份额
    priority_dic:各目标函数优先级
    MMN:max_market_num 高峰期约束下某个市场所能接的最大案子数
    market_case:此市场对应的case集合
    starttime:案子j的开始日期
    duration:案子j的持续时间
    INC:min_number_Case = {}  # 按份额计算出的最小接案数
    MNC:max_number_Case = {}  # 按份额计算出的最大接案数
    type_case_dic:各类型的案子集合
    contractor_property:厂商历史数据
    case_area:案子面积
    period_low:需要考虑的低峰月月份
    cur_low_num:已接低峰月案子数
    is_start:案子j是否在日期t开始
    is_ctime:案子j在日期t是否在施工
    current_num:厂商已接份额数
    is_new_year:是否为新年案
    cur_newyear_num:已接新年案数目
    contractor_market:厂商i可服务的市场集合
    key:新旧合同期
    area_total_num:此区域已接份额数

    '''
    model = gp.Model('Allocation_multiobj')

    # Initialize decision variables
    x = model.addVars(I, J, vtype=GRB.BINARY, name='is_dispatch')
    ns = model.addVars(I, vtype=GRB.BINARY, name='new_start_parameter')
    ny = model.addVars(I, vtype=GRB.BINARY, name='new_year_parameter')
    s_a = model.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS, name='sum_allot_case')
    s = model.addVar(lb=0, ub=1000, vtype=GRB.INTEGER, name='sum_case')
    s_type = model.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS, name='sum_case_type')
    a_1 = model.addVars(J, J, vtype=GRB.BINARY, name='abs_pre_parameter1')
    a_2 = model.addVars(J, K, vtype=GRB.BINARY, name='abs_pre_parameter2')
    update_casenum = model.addVars(I, lb=0, ub=1000, vtype=GRB.CONTINUOUS, name='update_share')

    # 每个案子只能由一个厂商接
    model.addConstrs(quicksum(x[i, j] for i in I) <= 1 for j in J)

    # 厂商与案子需匹配
    model.addConstrs(x[i, j] * (1 - ICC[i, j]) == 0 for i in I for j in J)

    # 同期最大施工量约束
    if maxnum_cntr:
        model.addConstrs(
            occupied[i, t] + quicksum(x[i, j] * is_ctime[j, t] for j in J) <= maxsamenum[i] for i in I for t in T)

    model.addConstr(s_a == quicksum(x[i, j] * type[j] for i in I for j in J))  # 此次派案总份额
    model.addConstr(s_type == s_a + area_total_num)  # 历史份额加本次派案总份额
    model.addConstr(s == quicksum(x[i, j] for i in I for j in J))  # 总接案数

    index = 0
    # 新合同期约束:
    if new_start_cntr:
        model.addConstrs(quicksum(x[i, j] for j in J) + current_num[i] + ns[i] >= 0.5 for i in I)
        model.setObjectiveN(quicksum(ns[i] * target_share[i] for i in I), index, priority=priority_dic['obj_new_start'],
                            abstol=1e-6,
                            reltol=0, name='obj_new_start')
        env0 = model.getMultiobjEnv(index)
        env0.setParam('MIPGap', gap_dic['obj_new_year'])
        env0.setParam(GRB.Param.NonConvex, 2)
        env0.setParam('PrePasses', 1)
        index += 1

    # 新年约束:
    if new_year_cntr:
        model.addConstrs(quicksum(x[i, j] * is_new_year[j] for j in J) + cur_newyear_num[i] + ny[i] >= 1 for i in I)
        model.setObjectiveN(quicksum(ny[i] * target_share[i] for i in I), index, priority=priority_dic['obj_new_year'],
                            abstol=1e-6,
                            reltol=0, name='obj_new_year')
        env1 = model.getMultiobjEnv(index)
        env1.setParam('MIPGap', gap_dic['obj_new_year'])
        env1.setParam(GRB.Param.NonConvex, 2)
        env1.setParam('PrePasses', 1)
        index += 1

    # 接案数尽可能多
    if case_num_cntr:
        model.setObjectiveN(len(J) - s, index, priority=priority_dic['obj_case_num'],
                            abstol=1e-6,
                            reltol=0, name='obj_case_num')
        env2 = model.getMultiobjEnv(index)
        env2.setParam('MIPGap', gap_dic['obj_case_num'])
        env2.setParam(GRB.Param.NonConvex, 2)
        env2.setParam('PrePasses', 1)
        index += 1

    # 区间gap
    if share_interval_gap_cntr:
        sp_1 = model.addVars(I, lb=0, ub=100, vtype=GRB.INTEGER, name='share_parameter1')
        sp_2 = model.addVars(I, lb=0, ub=100, vtype=GRB.INTEGER, name='share_parameter2')
        max_1 = model.addVar(ub=1000, vtype=GRB.CONTINUOUS, name='max_parameter_1')
        max_2 = model.addVars(I, lb=-1000, ub=1000, vtype=GRB.CONTINUOUS, name='max_parameter_2')
        inc = model.addVars(I, lb=-1000, ub=1000, vtype=GRB.CONTINUOUS, name='min_number_Case')
        mnc = model.addVars(I, lb=-1000, ub=1000, vtype=GRB.CONTINUOUS, name='max_number_Case')
        para_gap = model.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS, name='share_parameter_gap')
        model.addConstr(max_1 == ResetProp * s_type)
        model.addConstr(para_gap == max_(ResetQuanti, max_1))
        model.addConstrs(max_2[i] == s_type * target_share[i] + para_gap for i in I)
        model.addConstrs(inc[i] == s_type * target_share[i] - para_gap for i in I)
        model.addConstrs(mnc[i] == max_(0, max_2[i]) for i in I)
        model.addConstrs(quicksum(x[i, j] * type[j] for j in J) + current_num[i] + sp_1[i] * 0.5 >= inc[i] for i in I)
        model.addConstrs(quicksum(x[i, j] * type[j] for j in J) + current_num[i] - sp_2[i] * 0.5 <= mnc[i] for i in I)

        model.setObjectiveN(quicksum(sp_1[i] + sp_2[i] for i in I), index, priority=priority_dic['obj_interval_gap'],
                            abstol=1e-6,
                            reltol=0, name='obj_interval_gap')
        env3 = model.getMultiobjEnv(index)
        env3.setParam('MIPGap', gap_dic['obj_max_gap'])
        env3.setParam(GRB.Param.NonConvex, 2)
        env3.setParam('PrePasses', 1)
        index += 1

    #  重置案约束
    if reset_cntr:
        model.setObjectiveN(
            quicksum(x[i, j] * IR[j] * (1 if ((OC[j] in I) & (OC[j] != i)) else 0) for i in I for j in J), index,
            priority=priority_dic['obj_reset'],
            abstol=1e-6, reltol=0, name='obj_reset')
        env4 = model.getMultiobjEnv(index)
        env4.setParam('MIPGap', gap_dic['obj_reset'])
        env4.setParam(GRB.Param.NonConvex, 2)
        env4.setParam('PrePasses', 1)
        index += 1
    if address_cntr | city_cntr:
        model.addConstrs(a_2[j, k] >= x[i, j] - ICC_h[i, k] for j in J for k in K for i in I)
        model.addConstrs(a_2[j, k] >= ICC_h[i, k] - x[i, j] for j in J for k in K for i in I)

    # 同地址约束
    if address_cntr:
        model.addConstrs(a_1[j1, j2] >= x[i, j1] - x[i, j2] for j1 in J for j2 in J for i in I)
        model.setObjectiveN(quicksum((quicksum(a_1[j1, j2] * SE[j1, j2] for j2 in J) + quicksum(
            a_2[j1, k] * SE[j1, k] for k in K)) for j1 in J), index,
                            priority=priority_dic['obj_same_address'],
                            abstol=1e-6, reltol=0, name='obj_same_address')
        env5 = model.getMultiobjEnv(index)
        env5.setParam('MIPGap', gap_dic['obj_same_address'])
        env5.setParam(GRB.Param.NonConvex, 2)
        env5.setParam('PrePasses', 1)
        index += 1

        # 就近派案
    if min_distance_cntr:
        model.setObjectiveN(quicksum(distance_dic[i, j] * x[i, j] for i in I for j in J), index,
                            priority=priority_dic['obj_min_distance'],
                            abstol=1e-6, reltol=0, name='obj_min_distance')
        env6 = model.getMultiobjEnv(index)
        env6.setParam('MIPGap', gap_dic['obj_min_distance'])
        env6.setParam(GRB.Param.NonConvex, 2)
        env6.setParam('PrePasses', 1)
        index += 1

    if share_sum_gap_cntr | share_max_gap_cntr:
        share_gap = model.addVars(I, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='share_gap')
        share_gap_abs = model.addVars(I, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='share_gap_abs')
        updateshare = model.addVars(I, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='updateshare')
        model.addConstrs(updateshare[i] * s_type == current_num[i] + quicksum(x[i, j] * type[j] for j in J) for i in I)
        model.addConstrs(share_gap[i] == updateshare[i] - target_share[i] for i in I)
        model.addConstrs(share_gap_abs[i] == abs_(share_gap[i]) for i in I)

    # 最小化sum_share_gap
    if share_sum_gap_cntr:
        model.setObjectiveN(quicksum(share_gap_abs[i] for i in I), index, priority=priority_dic['obj_sum_share_gap'],
                            abstol=1e-6,
                            reltol=0, name='obj_sum_share_gap')
        env7 = model.getMultiobjEnv(index)
        env7.setParam('MIPGap', gap_dic['obj_sum_share_gap'])
        env7.setParam(GRB.Param.NonConvex, 2)
        env7.setParam('PrePasses', 1)
        index += 1

    # 极小化最大差异
    # 非线性份额
    if share_max_gap_cntr:
        model.addConstrs(update_casenum[i] == quicksum(x[i, j] * type[j] for j in J) for i in I)
        max_gap = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='max_gap')
        model.addConstrs(max_gap >= share_gap_abs[i] for i in I)
        model.setObjectiveN(max_gap, index, priority=priority_dic['obj_max_gap'], abstol=1e-6, reltol=0,
                            name='obj_max_gap')
        env8 = model.getMultiobjEnv(index)
        env8.setParam('MIPGap', gap_dic['obj_max_gap'])
        env8.setParam(GRB.Param.NonConvex, 2)
        env8.setParam('PrePasses', 1)
        index += 1

        # 线性份额
        # for i1 in I:
        #     for i2 in I:
        #         for j in J:
        #             model.addLConstr(g[i1, i2, j] <= x[i2, j])
        #             model.addLConstr(g[i1, i2, j] <= updateshare[i1])
        #             model.addLConstr(g[i1, i2, j] >= updateshare[i1] + x[i2, j] - 1)
        #
        # for i in I:
        #     model.addLConstr(
        #         updateshare[i] * area_total_num + quicksum(g[i, i0, j] * type[j] for i0 in I for j in J) ==
        #         current_num[i] + quicksum(x[i, j] * type[j] for j in J))

    # 同城市约束
    if city_cntr:
        model.setObjectiveN(quicksum(a_2[j, k] * SC[j, k] for j in J for k in K), index,
                            priority=priority_dic['obj_same_city'],
                            abstol=1e-6, reltol=0, name='obj_same_city')
        env9 = model.getMultiobjEnv(index)
        env9.setParam('MIPGap', gap_dic['obj_same_city'])
        env9.setParam(GRB.Param.NonConvex, 2)
        env9.setParam('PrePasses', 1)
        # env9.setParam('MIPFocus', 2)
        index += 1

    # 低峰月约束：
    if low_cntr:
        pl = model.addVars(I, vtype=GRB.BINARY, name='low_month_parameter')
        model.addConstrs(
            quicksum(x[i, j] * is_start[j, t] for t in p for j in J) + quicksum(cur_low_num[i, t] for t in p) + pl[
                i] >= 1
            for i in I for p in period_low)
        model.setObjectiveN(quicksum(pl[i] for i in I), index, priority=priority_dic['obj_low_month'],
                            abstol=1e-6,
                            reltol=0, name='obj_low_month')
        env10 = model.getMultiobjEnv(index)
        env10.setParam('MIPGap', gap_dic['obj_low_month'])
        env10.setParam(GRB.Param.NonConvex, 2)
        env10.setParam('PrePasses', 1)
        # env10.setParam('MIPFocus', 2)
        index += 1

    # Set global sense for ALL objectives
    model.ModelSense = GRB.MINIMIZE

    # Set Parameters
    # model.Params.OutputFlag=0
    model.setParam(GRB.Param.NonConvex, 2)
    model.Params.MIPGap = 0
    model.Params.PrePasses = 1
    model.Params.LogFile = path2 + '/{}合同期标段{}.log'.format(key, r)
    model.optimize()

    # Status checking
    status = model.Status
    if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        print("The model cannot be solved because it is infeasible or " " unbounded ")
        sys.exit(1)
    if status != GRB.OPTIMAL:
        print(' Optimization was stopped with status ' + str(status))
        sys.exit(1)

    # not_allot_num = 0

    for j in J:
        theline = {}
        theline['CaseCode'] = j
        if x.sum('*', j).getValue() == 0:
            theline['ContractorCode'] = ''
            theline['AllotLabel'] = '1'
            theline['AllotReason'] = 'MaxSameTime'
        else:
            for i in I:
                if x[i, j].X > 0.01:
                    theline['ContractorCode'] = i[:-7]
                    theline['AllotLabel'] = '0'
                    theline['AllotReason'] = ''
        jsresult.append(theline)

    mc = 0
    sheetname = 'region ' + str(r)
    result = pd.DataFrame(columns=['ContractorNameKey'])
    origshare = defaultdict(float)
    sum_case_num = s_type.x
    # result.loc[mc, 'not_allot_num'] = not_allot_num
    # result.loc[mc, 'sum_case'] = s.x
    for i in I:
        # print(i)
        if area_total_num != 0:
            origshare[i] = current_num[i] / area_total_num
        result.loc[mc, 'ContractorNameKey'] = i[:-7]
        result.loc[mc, 'TargetShare'] = target_share[i]
        result.loc[mc, 'OrigShare'] = origshare[i]
        result.loc[mc, 'OrigNum'] = current_num[i]
        result.loc[mc, 'CurrentShare'] = updateshare[i].x
        result.loc[mc, 'AllotCaseNum'] = update_casenum[i].x
        result.loc[mc, 'OrigShareGap'] = abs(round((target_share[i] - origshare[i]), 3))
        result.loc[mc, 'CurrentShareGap'] = abs(round((target_share[i] - updateshare[i].x), 3))
        result.loc[mc, 'CurrentNumGap'] = update_casenum[i].x + current_num[i] - target_share[i] * sum_case_num
        result.loc[mc, 'maxsamenum'] = maxsamenum[i]
        for k in T:
            if k == 0:
                continue
            result.loc[mc, str(nowday + datetime.timedelta(days=k - 1))[:10]] = occupied_0[i, k] + sum(
                x[i, j].x * is_ctime[j, k] for j in J)
        mc += 1

    result.to_excel(writer, sheet_name=sheetname)
    # worksheet = writer.sheets[sheetname]
    # for idx in range(result.shape[1] + 1):
    #     worksheet.set_column(idx, idx, 10)

    # except gp.GurobiError as e:
    #     print('Erroe code ' + str(e.errno) + ":" + str(e))
    # except AttributeError as e:
    #     print('Encountered an attribute error: ' + str(e))


#

for key in contractor_split.keys():
    contractorinfoall_temp = contractor_split.get(key)
    appliedcaseinfoall_temp = appliedcase_split.get(key)
    if appliedcaseinfoall_temp.empty or contractorinfoall_temp.empty:
        # print('此次派案中无{}合同期的案子'.format(key))
        continue
    writer = pd.ExcelWriter(path2 + '/result{}.xlsx'.format(key))

    '''

    厂商数据处理

    '''
    section_contractorcode_name = {}  # 按标段对厂商进行划分
    contractor_property = defaultdict(dict)  # 已服务的各类型店的数量信息
    maxsamenum = defaultdict(int)  # 厂商同期最大施工数
    target_share = defaultdict(float)  # 目标份额
    current_share = defaultdict(float)  # 当前份额
    current_num = defaultdict(int)  # 已接案子数
    section_total_num = defaultdict(int)  # 各标案子数（已接案）
    contractorcode_name = defaultdict(str)
    current_case_num = defaultdict(int)  # 不按份额计算的店数
    axis = defaultdict(list)
    contractorinfo = contractor_data_process(contractorinfoall=contractorinfoall_temp,
                                             contractorcode_name=contractorcode_name,
                                             section_contractorcode_name=section_contractorcode_name,
                                             contractor_property=contractor_property, current_case_num=current_case_num,
                                             maxsamenum=maxsamenum, axis=axis,
                                             target_share=target_share, current_share=current_share,
                                             current_num=current_num, section_list=section_list,
                                             section_total_num_h=section_total_num, )

    '''

    历史案数据处理

    '''

    occupied = defaultdict(int)  # 厂商在日期t的施工数
    section_casecodelist_h = defaultdict(list)  # 按区域划分的已派案集合
    allot_avail_h = defaultdict(int)  # 匹配信息
    cur_endday = defaultdict(int)  # 案子结束时间
    cur_newyear_num = defaultdict(int)  # 当前已接新年案子数
    cur_newstart_num = defaultdict(int)  # 当前已接新合同期案子数
    cur_low_num = defaultdict(int)  # 当前已接低峰月案子数
    currentcaseinfo = currentcase_data_process(currentcaseinfo=currentcaseinfoall, occupied=occupied,
                                               cur_newyear_num=cur_newyear_num,
                                               contractorcode_name=contractorcode_name, nowday=nowday,
                                               newyearstartday=newyearstartday,
                                               cur_low_num=cur_low_num,
                                               allot_avail_h=allot_avail_h, endday_dic=cur_endday,
                                               SameTimeLength=SameTimeLength,
                                               low_start_Day=low_start,
                                               low_end_day=low_end,
                                               ContractStartDay=ContractStartDay,
                                               region_casecodelist_h=section_casecodelist_h, isnew=key,
                                               cur_newstart_num=cur_newstart_num)
    '''

    当前案子数据处理

    '''
    isreset = defaultdict(int)  # 是否重置
    origcontractor = defaultdict(str)  # 原厂商
    allot_avail = defaultdict(int)  # 若当前case可被分配至厂商，那么allot_avail[contractorcode, case] = 1
    allot_availset = defaultdict(set)  # 当前case可分配的厂商集合
    allot_share_num = defaultdict(float)  # 按份额计算方法得到的各区域的案子数目
    allot_num = defaultdict(int)  # 此轮申请案子各区域的分布数量
    type_para = defaultdict(int)  # 待派案若为PIE案，则为0.5，否则为1,用于份额计算
    section_casecodelist = defaultdict(list)  # 按标段划分的待派案集合
    is_sameadress = defaultdict(int)  # 是否同地址
    is_samecity = defaultdict(int)  # 同城市参数
    starttime = defaultdict(int)  # 案子开始时间
    duration = defaultdict(int)  # 案子持续时间
    is_low = defaultdict(int)  # 是否为低峰月案子
    is_new_year = defaultdict(int)  # 是否为新年案
    distance_dic = defaultdict(int)
    case_axis = defaultdict(list)
    type_case_dic = defaultdict(set)
    is_ctime = defaultdict(int)
    max_day = [0]  # 此次派案最迟结束时间
    appliedcaseinfo = appliedcase_data_process(appliedcaseinfoall=appliedcaseinfoall_temp,
                                               jsresult=jsresult, currentcaseinfo=currentcaseinfo,
                                               nowday=nowday, nosamecity=nosamecity, case_axis=case_axis,
                                               newyearstartday=newyearstartday, distance_dic=distance_dic,
                                               type=type_para, starttime=starttime, duration=duration,
                                               max_day=max_day, cur_endday=cur_endday, is_low=is_low,
                                               contractorinfo=contractorinfo, allot_avail=allot_avail,
                                               allot_availset=allot_availset, section_list=section_list,
                                               region_casecodelist=section_casecodelist, axis=axis,
                                               low_start_day=low_start, low_end_day=low_end,
                                               SameTimeLength=SameTimeLength, is_new_year=is_new_year,
                                               isreset=isreset, is_samecity=is_samecity,
                                               is_sameadress=is_sameadress, allot_share_num=allot_share_num,
                                               OC=origcontractor, is_ctime=is_ctime, type_case_dic=type_case_dic,
                                               allot_num=allot_num)
    '''

        gurobi运行前的参数处理

    '''

    # 日期集合
    min_day = 0
    if key == 'new':
        min_day = (ContractStartDay - nowday).days + 1
    period = list(range(min_day, max_day[0] + 1))

    # 校验下目前厂商的同期接案量是否已经超过了最大接案量，如果超过了，则为最大值，这样保证既不会给厂商派案，也不会影响同期最大接案量的约束
    # exceed = defaultdict(list)
    occupied_deal = copy.deepcopy(occupied)
    for i in contractorcode_name.keys():
        for k in period:
            # if occupied[i,k] > maxsamenum[i]:
            #     exceed[i].append(k)
            if occupied_deal[i, k] > maxsamenum[i]:
                occupied_deal[i, k] = maxsamenum[i]

    # 偏差处理 section_total_num 可能要改成体量
    Prop = defaultdict(float)
    Quanti = defaultdict(int)
    for r in section_list:
        for ky in propdeviation.keys():
            if section_total_num[r] <= propdeviation[ky][1]:
                Prop[r] = float(ky)
                break
        for ky in quantideviation.keys():
            if section_total_num[r] <= quantideviation[ky][1]:
                Quanti[r] = int(ky)
                break

    # 新年约束
    if newyear_cntr_starttime <= nowday <= newyear_cntr_endtime:
        new_year_cntr = True
    # 低峰月约束
    period_low_list = list()
    period_low = tuple()
    if low_start - datetime.timedelta(days=start_low) <= nowday <= low_end - datetime.timedelta(days=end_low):
        low_cntr = True
        end = (low_end - nowday).days + 1
        if nowday <= low_start:
            start = (low_start - nowday).days + 1
        else:
            low_start_1 = low_start
            while nowday > low_start_1:
                low_start_1 = low_start_1 + datetime.timedelta(days=30)
            start = (low_start_1 - nowday).days + 1
        while (start + SameTimeLength < period[-1]) & (start < end):
            list_temp = list(range(start, min(start + 31, period[-1])))
            period_low_list.append(list_temp)
            start += 31
        period_low = [tuple(p) for p in period_low_list]
    # 分标段进行求解
    # print(distance_dic)
    for r in section_list:
        if not len(section_casecodelist[r]) == 0:
            gurobi_processing(I=section_contractorcode_name[r].keys(), J=section_casecodelist[r],
                              area_total_num=section_total_num[r], nowday=nowday,
                              K=section_casecodelist_h[r], T=period, r=r, type=type_para,
                              is_ctime=is_ctime, key=key, gap_dic=gap_dic, ResetProp=Prop[r],
                              maxsamenum=maxsamenum, occupied=occupied_deal, ResetQuanti=Quanti[r],
                              ICC=allot_avail, ICC_h=allot_avail_h, share_max_gap_cntr=share_max_gap_cntr,
                              IR=isreset, OC=origcontractor, SE=is_sameadress, occupied_0=occupied,
                              SC=is_samecity, target_share=target_share, is_start=is_low,
                              priority_dic=priority_dic, cur_newyear_num=cur_newyear_num,
                              jsresult=jsresult, cur_low_num=cur_low_num, current_num=current_num,
                              writer=writer, is_new_year=is_new_year, period_low=period_low,
                              new_start_cntr=new_start_cntr, new_year_cntr=new_year_cntr,
                              maxnum_cntr=maxnum_cntr, reset_cntr=reset_cntr,
                              address_cntr=address_cntr, share_sum_gap_cntr=share_sum_gap_cntr,
                              share_interval_gap_cntr=share_interval_gap_cntr,
                              city_cntr=city_cntr, low_cntr=low_cntr, distance_dic=distance_dic)
    writer.close()

'''

文件输出

'''

with open('out/AllotResult.json', "w") as jsfile:
    json.dump(jsresult, jsfile)

allend = time.time()
file = open(path2 + '/ResultLog.txt', 'w')
file.write('Total time :' + str(allend - allstart) + '\n')
file.write('Today: ' + str(today) + '\n')
print('Algorithm Total Time: ' + str(allend - allstart))
file.close()

'''

未分配案子统计

'''

allot_result = pd.read_json('out/AllotResult.json')
id_nonallot = allot_result[allot_result["AllotLabel"] == 1].index
if len(id_nonallot) == 0:
    pass
else:
    writer_1 = pd.ExcelWriter('log/non_allot_result.xlsx')
    non_allot_result = pd.DataFrame(columns=['CaseCode'])
    mc = 0
    stl = int(SameTimeLength)
    for id in id_nonallot:
        casecode = allot_result.loc[id, 'CaseCode']
        non_allot_result.loc[mc, 'CaseCode'] = casecode
        non_allot_result.loc[mc, 'nonAllotReason'] = allot_result.loc[id, 'AllotReason']
        if not allot_availset[casecode]:
            non_allot_result.loc[mc, '可分配厂商'] = 'None'
        else:
            non_allot_result.loc[mc, '可分配厂商'] = str(allot_availset[casecode])
        starttime_str = str(appliedcaseinfoall[appliedcaseinfoall["CaseCode"] == casecode].ApproachDate.values[0])[:10]
        non_allot_result.loc[mc, 'ApproachDate'] = starttime_str
        case_type = appliedcaseinfoall[appliedcaseinfoall['CaseCode'] == casecode].CaseType.values[0]
        if case_type == 1 or case_type == 2:
            non_allot_result.loc[mc, 'CompletionDate'] = str(
                datetime.datetime.strptime(starttime_str, '%Y/%m/%d') + datetime.timedelta(days=stl - 1))[:10]
        else:
            non_allot_result.loc[mc, 'CompletionDate'] = str(
                appliedcaseinfoall[appliedcaseinfoall["CaseCode"] == casecode].CompletionDate.values[0])[:10]
        mc += 1
    non_allot_result.to_excel(writer_1)
    # worksheet = writer_1.sheets["Sheet1"]
    # for idx in range(non_allot_result.shape[1] + 1):
    #     if idx == 0:
    #         continue
    #     worksheet.set_column(idx, idx, 15)
    writer_1.close()
