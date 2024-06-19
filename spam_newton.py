import numpy as np #수학적 연산을 위한 패키지


#시그모이드 함수(입력값을 0~1로 변환)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#단어 리스트에서 n-그램 생성
def generate_ngrams(words, n):
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(' '.join(words[i:i + n]))  # n-그램 생성
    return ngrams


#학습 데이터에서 유니그램 및 바이그램 특징 추출, 스팸 단어 리스트를 기반으로 벡터화
def extract_features(email, spam_words):
    words = email.split()
    features_unigram = [1 if word in words else 0 for word in spam_words]
    features_bigram_list = generate_ngrams(words, 2)
    features_bigram = [1 if bigram in features_bigram_list else 0 for bigram in spam_words]
    features = features_unigram + features_bigram  # 유니그램과 바이그램 특징 결합
    return features


#학습 데이터와 스팸 단어 리스트를 기반으로 특징 벡터와 레이블 생성
def prepare_data(emails, spam_words):
    X = []
    y = []
    for email, label in emails:
        features = extract_features(email, spam_words)  # 특징 추출
        X.append(features)  # 특징 벡터 추가
        y.append(label)  # 레이블 추가

    X = np.array(X)  # 리스트를 배열로 변환
    y = np.array(y)

    return X, y


#뉴턴-랩슨법을 이용한 로지스틱 회귀 모델 학습
def newton_method_logistic_regression(X, y, max_iter=100, tol=1e-6, lambda_reg=1e-4):
    m, n = X.shape  # 샘플 수와 특징 수

    weights = np.zeros(n)  # 가중치 초기화
    best_loss = np.inf  # 최적의 손실 값 초기화
    best_iter = 0  # 최적의 반복 초기화

    for iteration in range(max_iter):
        y_hat = sigmoid(np.dot(X, weights))  # 예측값 계산
        gradient = np.dot(X.T, y_hat - y) / m  # 그래디언트 계산
        R = np.diag(y_hat * (1 - y_hat))  # 헤시안 행렬의 대각 성분 계산
        H = np.dot(X.T, np.dot(R, X)) / m  # 헤시안 행렬 계산
        H += lambda_reg * np.eye(n)  # 정규화를 통한 안정화

        try:
            delta = np.linalg.solve(H, gradient)  # 가중치 업데이트 계산
        except np.linalg.LinAlgError:
            print("Hessian is singular, stopping early")
            break

        weights -= delta  # 가중치 업데이트

        loss = compute_loss(y, sigmoid(np.dot(X, weights)))  # 손실 계산

        if loss < best_loss:  # 최적의 손실 값 업데이트
            best_loss = loss
            best_iter = iteration
        elif iteration - best_iter > 10:  # 조기 종료 조건
            print(f'Early stopping at iteration {iteration}')
            break

        if iteration % 10 == 0:  # 10번째 반복마다 손실 출력
            print(f'Iteration {iteration}, Loss: {loss}')

        if np.linalg.norm(delta, ord=1) < tol:  # 수렴 조건 확인
            print(f'Convergence achieved at iteration {iteration}')
            break

    return weights


#특징 데이터에 대한 예측 확률 계산. 예측값이 0.5보다 크면 1(스팸), 작으면 0(정상)
def predict(X, weights):
    y_hat = sigmoid(np.dot(X, weights))
    return y_hat > 0.5


#로지스틱 회귀 모델의 손실 값 계산 (교차 엔트로피)
def compute_loss(y, y_hat):
    m = len(y)
    loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
    return loss


#학습 데이터와 스팸 단어 리스트를 사용하여 모델 학습 및 평가
def main(emails, spam_words, max_iter=100, tol=1e-6):
    X, y = prepare_data(emails, spam_words)  # 데이터 준비

    weights = newton_method_logistic_regression(X, y, max_iter, tol)  # 뉴턴-랩슨법 사용

    y_hat = predict(X, weights)  # 예측
    accuracy = np.mean(y_hat == y)  # 정확성 계산
    precision = np.mean(y_hat[y == 1] == 1) if np.sum(y == 1) > 0 else 0  # 정밀도 계산
    recall = np.mean(y[y_hat == 1] == 1) if np.sum(y_hat == 1) > 0 else 0  # 재현율 계산
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  # F1 점수 계산

    print(f'정확성: {accuracy:.4f}')
    print(f'정밀도: {precision:.4f}')
    print(f'재현율: {recall:.4f}')
    print(f'F1 점수: {f1_score:.4f}')
    return weights


# 학습 데이터
emails = [
    ("Win a free vacation now!", 1),
    ("Your account has been compromised, please reset your password", 1),
    ("Meeting tomorrow at 10 AM", 0),
    ("Limited time offer, buy one get one free", 1),
    ("Project discussion rescheduled", 0),
    ("Congratulations! You've won a $1000 gift card", 1),
    ("Weekly report attached", 0),
    ("Claim your free prize today", 1),
    ("Lunch meeting on Friday", 0),
    ("Exclusive deal just for you", 1),
    ("Family reunion this weekend", 0),
    ("Team outing next month", 0),
    ("Huge discount on our new products", 1),
    ("Company policy update", 0),
    ("Earn money from home easily", 1),
    ("Happy Birthday! Enjoy your special day", 0),
    ("Free trial of our new service", 1),
    ("Don't miss our annual sale", 1),
    ("Please review the attached document", 0),
    ("Special promotion just for you", 1),
    ("Staff meeting this Thursday", 0),
    ("Check out these new features", 0),
    ("Thank you for your purchase", 0),
    ("Act now and save 50%", 1),
    ("Your invoice is attached", 0),
    ("Get rich quick scheme", 1),
    ("Join us for the webinar", 0),
    ("Exclusive offer just for you", 1),
    ("New security alert for your account", 1),
    ("Upcoming holiday schedule", 0),
    ("Final reminder: Payment due", 1),
    ("Company-wide meeting tomorrow", 0),
    ("Hot deals on electronics", 1),
    ("Invitation to connect on LinkedIn", 0),
    ("Claim your free trial now", 1),
    ("Quarterly financial results", 0),
    ("Urgent: Verify your email address", 1),
    ("Next month's newsletter", 0),
    ("Get your free eBook now", 1),
    ("Client meeting rescheduled", 0),
    ("Earn extra income online", 1),
    ("Weekly team sync-up", 0),
    ("Exclusive savings on your favorite items", 1),
    ("Immediate action required", 1),
    ("Holiday party next week", 0),
    ("Special offer ends soon", 1),
    ("Important update about your order", 1),
    ("Team lunch this Friday", 0),
    ("Hurry, limited time offer", 1),
    ("Your order has been shipped", 0),
    ("Discover our new arrivals", 1),
    ("Monthly performance review", 0),
    ("Click here to claim your prize", 1),
    ("Upcoming project deadlines", 0),
    ("Free membership upgrade", 1),
    ("Birthday celebration next Monday", 0),
    ("Please complete the survey", 0),
    ("Save big with our exclusive offer", 1),
    ("Employee of the month announcement", 0),
    ("Act now for a special bonus", 1),
    ("Team-building event next week", 0),
    ("Win an iPhone for free", 1),
    ("Monthly budget meeting", 0),
    ("Last chance to save 40%", 1),
    ("Your payment has been received", 0),
    ("Try our new product for free", 1),
    ("Don't miss out on this offer", 1),
    ("Year-end performance review", 0),
    ("Free gift with purchase", 1),
    ("Schedule change notice", 0),
    ("Congratulations! You've won", 1),
    ("Weekly progress report", 0),
    ("Get your free sample now", 1),
    ("Board meeting next Tuesday", 0),
    ("Office renovation next month", 0),
    ("Staff outing this weekend", 0),
    ("Claim your free gift now", 1),
    ("Monthly sales update", 0),
    ("Special discount for loyal customers", 1),
    ("Security alert: Suspicious activity", 1),
    ("Training session rescheduled", 0),
    ("Flash sale ends tonight", 1),
    ("Your package is on its way", 0),
    ("Win a trip to Hawaii", 1),
    ("Team meeting rescheduled", 0),
    ("Limited time: Get a free trial", 1),
    ("Project milestone achieved", 0),
    ("Exclusive offer: Act now", 1),
    ("Holiday greetings from our team", 0),
    ("Get a free consultation", 1),
    ("New policy updates", 0),
    ("Don't miss your chance to win", 1),
    ("Weekly team meeting", 0),
    ("당신의 컴퓨터가 바이러스에 걸렸습니다", 1),
    ("내일 우량주를 무료로 수령하세요", 1),
    ("저한테 연락하셔서 교수님의 생방송 수업을 예약하세요", 1),
    ("채팅방 참여시 매일 10% 단타 종목 출첵시 치킨 지급", 1),
    ("코스닥 상장 확정 종목", 1),
    ("50% 할인 행사 마감 1시간 전", 1),
    ("Your account has benn hacked", 1),
    ("무료 휴가를 지금 받으세요!", 1),
    ("귀하의 계정이 손상되었습니다. 비밀번호를 재설정하십시오.", 1),
    ("내일 오전 10시에 회의가 있습니다.", 0),
    ("한정된 시간 동안, 하나 사면 하나 무료", 1),
    ("프로젝트 논의가 재조정되었습니다.", 0),
    ("축하합니다! $1000 기프트 카드를 받으셨습니다.", 1),
    ("주간 보고서가 첨부되었습니다.", 0),
    ("지금 무료 상품을 청구하세요", 1),
    ("금요일 점심 회의", 0),
    ("당신을 위한 독점 거래", 1),
    ("이번 주말에 가족 모임", 0),
    ("긴급: 결제 정보를 업데이트하세요", 1),
    ("다음 달 팀 아웃팅", 0),
    ("새 제품에 대한 대규모 할인", 1),
    ("회사 정책 업데이트", 0),
    ("집에서 쉽게 돈 벌기", 1),
    ("생일 축하합니다! 특별한 하루 되세요", 0),
    ("우리의 새로운 서비스를 무료로 시도하세요", 1),
    ("연례 세일을 놓치지 마세요", 1),
    ("첨부된 문서를 검토해주세요", 0),
    ("당신을 위한 특별 프로모션", 1),
    ("이번 목요일 직원 회의", 0),
    ("이 새로운 기능을 확인하세요", 0),
    ("구매해 주셔서 감사합니다", 0),
    ("지금 행동하여 50%를 절약하세요", 1),
    ("송장이 첨부되었습니다", 0),
    ("빠른 부자 되기 계획", 1),
    ("웹 세미나에 참여하세요", 0),
    ("당신을 위한 독점 제안", 1),
    ("귀하의 계정에 대한 새로운 보안 경고", 1),
    ("다가오는 휴일 일정", 0),
    ("최종 알림: 결제 기한", 1),
    ("내일 회사 전체 회의", 0),
    ("전자 제품에 대한 핫 딜", 1),
    ("LinkedIn에서 연결 초대", 0),
    ("지금 무료 평가판을 청구하세요", 1),
    ("분기별 재무 결과", 0),
    ("빠르게 행동하세요: 제한된 재고", 1),
    ("긴급: 이메일 주소를 확인하세요", 1),
    ("다음 달 뉴스레터", 0),
    ("무료 eBook을 지금 받으세요", 1),
    ("클라이언트 회의가 재조정되었습니다", 0),
    ("온라인에서 추가 수익을 올리세요", 1),
    ("주간 팀 싱크업", 0),
    ("당신이 좋아하는 항목에 대한 독점 절약", 1),
    ("즉각적인 조치가 필요합니다", 1),
    ("다음 주 휴일 파티", 0),
    ("특별 제안이 곧 종료됩니다", 1),
    ("주문에 대한 중요한 업데이트", 1),
    ("이번 금요일 팀 점심", 0),
    ("주문이 배송되었습니다", 0),
    ("월간 성과 검토", 0),
    ("상품을 청구하려면 여기를 클릭하세요", 1),
    ("다가오는 프로젝트 마감일", 0),
    ("무료 멤버십 업그레이드", 1),
    ("다음 월요일 생일 축하 파티", 0),
]

# 스팸 단어 리스트
spam_words = [
    "free", "win", "money", "lottery", "now", "offer", "limited", "buy", "discount", "earn",
    "cash", "urgent", "important", "prize", "gift", "claim", "trial", "save", "click", "act",
    "congratulations", "selected", "winner", "big", "best", "cheap", "clearance", "credit", "deal",
    "exclusive", "extra", "fantastic", "guarantee", "increase", "instant", "investment", "miracle",
    "promise", "promotion", "reward", "rich", "risk", "special", "unlimited", "verify", "warning",
    "bonus", "dear", "easy", "fast", "great", "income", "luxury", "million", "online", "profit",
    "quick", "satisfaction", "wow", "bargain", "discount", "hack", "hacked", "danger", "긴급", "행운", "무료", "당첨", "돈", "지금", "제안", "한정된", "구매", "할인", "벌다", "현금", "중요한", "상", "선물", "청구", "시도", "저장", "클릭", "행동", "축하",
    "승자", "크다", "최고의", "싼", "정리", "신용", "거래", "독점적인", "추가", "환상적인", "보증", "증가",
    "즉시", "투자", "기적", "약속", "홍보", "보상", "부유한", "위험", "특별한", "무제한", "긴급한", "확인",
    "경고", "승리", "친애하는", "쉬운", "선물", "크다", "증가", "수입", "사치", "백만", "온라인", "이익",
    "기회", "amazing", "apply", "bargain", "benefits", "best price", "big savings", "blockbuster", "breakthrough",
    "certified", "congratulations", "credits", "deal of the day", "deluxe", "double your", "earn money",
    "exclusive deal", "free bonus", "free access", "free demo", "free download", "free gift", "free info",
    "free membership", "free preview", "free quote", "free trial", "free website", "get paid", "gift card",
    "gift certificate", "giveaway", "great offer", "guarantee", "incredible deal", "increase sales",
    "insurance", "join free", "low price", "luxury", "make money", "miracle cure", "no obligation", "no risk",
    "offer expires", "one time", "promise you", "real thing", "reduce debt", "risk-free", "super", "chat", "telegram",
    "공짜", "고소득", "기회", "긴급", "돈벌기", "드림카", "무상", "무이자", "보너스", "무료체험", "무료배송", "무료설치",
    "무조건", "백만장자", "부자되기", "빨리", "상담예약", "성공", "세일", "소득증가", "신속", "신용", "쌀 때", "알뜰", 
    "안심", "이자", "이익", "인증", "자격", "저렴", "절약", "정말", "좋은 조건", "최고", "최저가", "특가", "특별제공", 
    "투자", "확실한", "할인", "혜택", "환상", "희귀", "획득", "주식", "단타", "코스닥", "채팅", "밴드", "텔레그램", "종목", "바이러스", "해킹", "http"
]


# 모델 학습 및 평가
weights = main(emails, spam_words)

test_emails = [
    "win big prizes",
    "your today schedule",
    "고객센터 문의 결과",
    "무료 투자 강의",
    "hello?",
    "안녕하세요",
    "무료 선물",
    "당신의 컴퓨터가 해킹되었습니다",
    "지금 시작시 무료 코인 지급",
    "무료 생방송 수업을 예약하세요",
    "채팅방 입장시 코스닥 상장 확정 종목 공개",
    "70% 할인 마감 1일 전",
    "Your computer has been hacked",
    "Report for exam",
    "Hello teacher",
    "new clothes discount only today"
]

#테스트 데이터 특징벡터로 변환
def prepare_test_data(test_emails, spam_words):
    X_test = []
    for email in test_emails:
        features = extract_features(email, spam_words)  # 특징 추출
        X_test.append(features)
    return np.array(X_test)




X_test = prepare_test_data(test_emails, spam_words)  # 테스트 데이터 준비
predictions = predict(X_test, weights)  # 테스트 데이터 예측

for i, email in enumerate(test_emails):
    print(f'{i+1}. "{email}": {"Spam" if predictions[i] else "Normal"}')  # 예측 결과 출력
