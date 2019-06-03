import xlrd
import xlwt
import heapq

# 原始数据中第一列为按键类型，其余各列为 N = 0~324的FFT

origin_workbook = xlrd.open_workbook(r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\data\train.xlsx')
origin_sheet = origin_workbook.sheet_by_index(0)
origin_sample_N = origin_sheet.nrows
origin_fft_N = origin_sheet.ncols - 1

peak_point = 30         # 只提取peak_point个峰
after_process_workbook = xlwt.Workbook(encoding='utf8')
after_sheet = after_process_workbook.add_sheet(u'sheet1')

for i in range(origin_sample_N):
    Keytype, *fft_list = origin_sheet.row_values(i)
    peak_list = heapq.nlargest(peak_point, fft_list)
    argmax_list = [fft_list.index(peak) for peak in peak_list]
    # 利用in判断这里可能会出现判断浮点数相等的问题
    argmax_list.insert(0, Keytype)
    for j in range(peak_point+1):
        after_sheet.write(i, j, argmax_list[j])

after_process_workbook.save(r'C:\Users\QinJingChang\PycharmProjects\Keyborad Voice Reconize\data\train_peak_arg.xls')
print("Run Completed")
