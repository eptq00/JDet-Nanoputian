## Source
Interview questions for VCIP, CS, Nankai University in the summer of 2025.<br>
paper link: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.pdf

## Install
For implementation details, see README-JDet.md.

## Train
        python JDet/tools/run_net.py --config-file=JDet\configs\lsknet-s_fpn_1x_dota_with_flip_my_torch2jittor.py --task=train
## Test
If you want to test the downloaded trained models, please set resume_path={you_checkpointspath} in the last line of the config file. 

        python JDet/tools/run_net.py --config-file=JDet\configs\lsknet-s_fpn_1x_dota_with_flip_my_torch2jittor.py --task=test

## Result
### Pytorch-ORCNN-DOTA v1.0 ss_split
(Data from ORCNN(ICCV 2021))<br> 

|MMRotate      |PL    |BD    |BR    |GTF   |SV    |LV    |SH    |TC    |BC    |ST    |SBF   |RA    |HA    |SP    |HC    |mAP    |
|:------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:-----:|
|ORCNN R-50    |89.46 |82.12 |54.78 |70.86 |78.93 |83.00 |88.20 |90.90 |87.50 |84.68 |63.97 |67.69 |74.94 |68.84 |52.28 |75.87  |


(Data from myself)<br> 
|MMRotate      |PL    |BD    |BR    |GTF   |SV    |LV    |SH    |TC    |BC    |ST    |SBF   |RA    |HA    |SP    |HC    |mAP    |
|:------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:-----:|
|ORCNN R-50    |89.26 |82.82 |53.17 |82.20 |78.62 |82.40 |87.96 |90.90 |86.24 |84.94 |64.32 |63.92 |67.91 |68.91 |53.22 |75.12  |

<img width="1593" height="580" alt="image" src="https://github.com/user-attachments/assets/5e96ca14-c892-489f-838c-85d6066f2964" />

### Jittor-ORCNN-DOTA v1.0 ss_split
(Data from myself)<br> 

|JDet          |PL    |BD    |BR    |GTF   |SV    |LV    |SH    |TC    |BC    |ST    |SBF   |RA    |HA    |SP    |HC    |mAP    |
|:------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:-----:|
|ORCNN R-50    |89.65 |82.70 |51.86 |73.22 |78.76 |77.88 |87.98 |90.90 |84.28 |85.63 |58.70 |65.61 |73.44 |69.32 |57.37 |75.15  |

<img width="1585" height="580" alt="image" src="https://github.com/user-attachments/assets/1bc9d596-b7f7-4616-a68d-62695a3feba5" />

log:    https://pan.baidu.com/s/1azEYd5MX5nX1qljPJ0bZgQ 提取码: 6efk  <br> 
model:  https://pan.baidu.com/s/1aTxY2hinpaaLl4iRjuutrQ 提取码: m3w3  <br> 

### Pytorch-LSKNet-DOTA v1.0 ss_split
(Data from PKINet(CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/papers/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.pdf)<br> 

|MMRotate      |PL    |BD    |BR    |GTF   |SV    |LV    |SH    |TC    |BC    |ST    |SBF   |RA    |HA    |SP    |HC    |mAP    |
|:------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:-----:|
|LSKNet-S      |89.66 |85.52 |57.72 |75.70 |74.95 |78.69 |88.24 |90.88 |86.79 |86.38 |66.92 |63.77 |77.77 |74.47 |64.82 |77.49  |

### Jittor-LSKNet-DOTA v1.0 ss_split
(Data from open-source code)<br>
|JDet-os       |PL    |BD    |BR    |GTF   |SV    |LV    |SH    |TC    |BC    |ST    |SBF   |RA    |HA    |SP    |HC    |mAP    |
|:------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:-----:|
|LSKNet-S      |89.79 |84.35 |56.22 |74.42 |74.73 |84.92 |88.56 |90.90 |87.05 |85.37 |63.16 |64.06 |77.20 |72.79 |58.44 |76.80  |
<img width="1596" height="587" alt="image" src="https://github.com/user-attachments/assets/b7d5503e-d8a6-4ee5-ad5a-eb6ec34a3f7b" />
log:    https://pan.baidu.com/s/1PE-h2adSZLG-3skfopNq2A 提取码: eicy <br>
model:  https://pan.baidu.com/s/1F_gdrI2iSwZ-u1DEKbG6Og 提取码: 675r <br>

(Data from myself) -Jittor v1-<br>
|JDet-myself   |PL    |BD    |BR    |GTF   |SV    |LV    |SH    |TC    |BC    |ST    |SBF   |RA    |HA    |SP    |HC    |mAP    |
|:------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:-----:|
|LSKNet-S      |89.87 |83.31 |54.32 |74.50 |78.67 |84.52 |88.74 |90.90 |86.98 |85.96 |60.47 |63.36 |77.06 |70.77 |68.25 |77.18  |

<img width="1593" height="598" alt="image" src="https://github.com/user-attachments/assets/4775356a-51f0-49a0-9f95-2d4b574bc25b" />

log:    https://pan.baidu.com/s/1upgeLpqV1q5VaVsRsvg2Kg 提取码: 9hx4  <br> 
model:  https://pan.baidu.com/s/1_7b3TXgarERhMlwce0lbjw 提取码: tvtm  <br> 

(Data from myself) -Jittor v2-<br>
|JDet-myself   |PL    |BD    |BR    |GTF   |SV    |LV    |SH    |TC    |BC    |ST    |SBF   |RA    |HA    |SP    |HC    |mAP    |
|:------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:-----:|
|LSKNet-S      |89.71 |85.00 |56.05 |78.48 |74.90 |84.47 |88.78 |90.90 |87.41 |86.51 |61.06 |64.01 |77.89 |71.84 |65.77 |77.52  |
<img width="1588" height="573" alt="image" src="https://github.com/user-attachments/assets/33128fe9-2a19-4637-8ef6-15b0902be7fe" />


log:https://pan.baidu.com/s/18QxURJXC71Q9yi5wxxkV8w 提取码: p9dv<br> 
model:https://pan.baidu.com/s/1hVwK36ClXdySca52ykpG8Q 提取码: bv39<br> 

