from durable.lang import *

with ruleset('Company_Rule'):
    @when_all((m.predicate == "라인성 불량이 있으면") & (m.do == "이미지 센서를 교체한다."))
    def Rule_1(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "Defect Pixel이 있으면") & (m.do == "Defect Pixel Correction을 진행 한다."))
    def Rule_2(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.subject == "Defect Correction을 진행 했다") & (m.predicate == "Defect Pixel Correction이 되지않는다.") & (m.do == "이미지 센서를 교체한다."))
    def Rule_3(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "연결 되지 않는다.") & (m.do == "카메라 전원 상태를 확인 한다."))
    def Rule_4(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.subject == "카메라가 연결 되지 않는다.") & (m.predicate == "카메라 전원 상태가 정상이다.") & (m.do == "카메라의 LED Indicator를 보고 확인 한다."))
    def Rule_5(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "상태가 비정상이다.") & (m.do == "카메라와 USB to Serial 연결을 하여 부팅메세지를 확인 한다."))
    def Rule_6(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "USB to Serial 연결이 되지 않는다.") & (m.do == "카메라와 USB to Serial 연결상태를 확인 한다."))
    def Rule_7(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.subject == "카메라의 LED Indicator 상태가 비정상이다.") & (m.predicate == "카메라와 USB to Serial이 연결 되었다.") & (m.do == "부팅 메세지를 확인 한다."))
    def Rule_8(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "Error 메세지가 있다.") & (m.do == "Error 메세지 부분을 디버깅 한다."))
    def Rule_9(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "제어가 되지 않는다.") & (m.do == "Serial 연결을 확인 한다."))
    def Rule_10(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.subject == "Serial 연결이 정상이다.") & (m.predicate == "제어가 되지 않는다.") & (m.do == "제어 커맨드를 확인 한다."))
    def Rule_11(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.subject == "제어 커맨드 확인 했다.") & (m.predicate == "제어 커맨드가 틀리다.") & (m.do == "제어 커맨드를 수정 한다."))
    def Rule_12(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.subject == "카메라 전원이 들어오지 않는다.") & (m.predicate == "카메라가 PoCXP를 사용하는지 확인 한다.") & (m.do == "Grabber의 PoCXP커넥터의 전원 연결을 확인 한다."))
    def Rule_13(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do + "or 외부전원 사용 카메라의 결우 외부전원을 인가 한다."})

    @when_all((m.predicate == "영상이 Grab 되지 않는다.") & (m.do == "카메라 트리거 모드를 확인 환다."))
    def Rule_14(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "트리거 모드 확인 결과 On 이다.") & (m.do == "트리거 모드를 Off로 변경 한다."))
    def Rule_15(c):
            c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "트리거 모드를 On으로 사용 해야 한다.") & (m.do == "Grabber의 트리거 관련 설정을 한다."))
    def Rule_16(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "온도가 정상 범위가 아니다.") & (m.do == "카메라의 Fan 동작을 확인 한다."))
    def Rule_17(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "온도가 정상 범위가 아니다. TEC Model 이다.") & (m.do == "카메라의 Fan 동작을 확인 한다. Peltier 동작을 확인 한다."))
    def Rule_18(c):
            c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "영상이 정상적이지 않다.") & (m.do == "카메라와 Grabber의 Image Bitdepth를 확인 한다."))
    def Rule_19(c):
        c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    @when_all((m.predicate == "Bayer 변환 후 색상이 상이하다.") & (m.do == "Bayer Conversion시 이미지 센서의 Bayer Pattern에 맞게 변환 하였는지 확인 한다."))
    def Rule_20(c):
            c.assert_fact({'subject': c.m.subject, 'predicate': c.m.predicate, 'do': c.m.do})

    # 규칙 출력
    @when_all(+m.subject)  # m.subject가 한번 이상
    def output(c):
        print('Fact: {0} {1} {2}'.format(c.m.subject, c.m.predicate, c.m.do))


assert_fact('Company_Rule', {'subject': '이미지 센서에', 'predicate': '라인성 불량이 있으면', 'do': '이미지 센서를 교체한다.'})
assert_fact('Company_Rule', {'subject': '이미지 센서에', 'predicate': 'Defect Pixel이 있으면', 'do': 'Defect Pixel Correction을 진행 한다.'})
assert_fact('Company_Rule', {'subject': 'Defect Correction을 진행 했다', 'predicate': 'Defect Pixel Correction이 되지않는다.', 'do': '이미지 센서를 교체한다.'})
assert_fact('Company_Rule', {'subject': '카메라가', 'predicate': '연결 되지 않는다.', 'do': '카메라 전원 상태를 확인 한다.'})
assert_fact('Company_Rule', {'subject': '카메라가 연결 되지 않는다.', 'predicate': '카메라 전원 상태가 정상이다.', 'do': '카메라의 LED Indicator를 보고 확인 한다.'})
assert_fact('Company_Rule', {'subject': '카메라 LED Indicator', 'predicate': '상태가 비정상이다.', 'do': '카메라와 USB to Serial 연결을 하여 부팅메세지를 확인 한다.'})
assert_fact('Company_Rule', {'subject': '카메라와', 'predicate': 'USB to Serial 연결이 되지 않는다.', 'do': '카메라와 USB to Serial 연결상태를 확인 한다.'})
assert_fact('Company_Rule', {'subject': '카메라의 LED Indicator 상태가 비정상이다.', 'predicate': '카메라와 USB to Serial이 연결 되었다.', 'do': '부팅 메세지를 확인 한다.'})
assert_fact('Company_Rule', {'subject': '부팅메세지에', 'predicate': 'Error 메세지가 있다.', 'do': 'Error 메세지 부분을 디버깅 한다.'})
assert_fact('Company_Rule', {'subject': 'DC Power Supply', 'predicate': '제어가 되지 않는다.', 'do': 'Serial 연결을 확인 한다.'})
assert_fact('Company_Rule', {'subject': 'Serial 연결이 정상이다.', 'predicate': '제어가 되지 않는다.', 'do': '제어 커맨드를 확인 한다.'})
assert_fact('Company_Rule', {'subject': '제어 커맨드 확인 했다.', 'predicate': '제어 커맨드가 틀리다.', 'do': '제어 커맨드를 수정 한다.'})
assert_fact('Company_Rule', {'subject': '카메라 전원이 들어오지 않는다.', 'predicate': '카메라가 PoCXP를 사용하는지 확인 한다.', 'do': 'Grabber의 PoCXP커넥터의 전원 연결을 확인 한다.'})
assert_fact('Company_Rule', {'subject': '카메라', 'predicate': '영상이 Grab 되지 않는다.', 'do': '카메라 트리거 모드를 확인 환다.'})
assert_fact('Company_Rule', {'subject': '카메라', 'predicate': '트리거 모드 확인 결과 On 이다.', 'do': '트리거 모드를 Off로 변경 한다.'})
assert_fact('Company_Rule', {'subject': '카메라', 'predicate': '트리거 모드를 On으로 사용 해야 한다.', 'do': 'Grabber의 트리거 관련 설정을 한다.'})
assert_fact('Company_Rule', {'subject': '이미지 센서', 'predicate': '온도가 정상 범위가 아니다.', 'do': '카메라의 Fan 동작을 확인 한다.'})
assert_fact('Company_Rule', {'subject': '이미지 센서', 'predicate': '온도가 정상 범위가 아니다. TEC Model 이다.', 'do': '카메라의 Fan 동작을 확인 한다. Peltier 동작을 확인 한다.'})
assert_fact('Company_Rule', {'subject': '카메라', 'predicate': '영상이 정상적이지 않다.', 'do': '카메라와 Grabber의 Image Bitdepth를 확인 한다.'})
assert_fact('Company_Rule', {'subject': 'Color 영상이', 'predicate': 'Bayer 변환 후 색상이 상이하다.', 'do': 'Bayer Conversion시 이미지 센서의 Bayer Pattern에 맞게 변환 하였는지 확인 한다.'})