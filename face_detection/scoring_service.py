import json
import math
import os

class ScoringService:
    def __init__(self, max_history=50, recent_fraction=0.2, recent_weight=2.0):
        self.raw_scores = []
        self.max_history = max_history
        self.recent_fraction = recent_fraction
        self.recent_weight = recent_weight
        self.current_question_index = 0
        self.has_shown_greeting = False
        
        # Map emotions to valence scores
        self.valence_map = {
            "Happiness": 1.0,
            "Surprise": 1.0,
            "Neutral": 0.0,
            "Sadness": -1.0,
            "Anger": -1.0,
            "Fear": -1.0,
            "Disgust": -1.0
        }
        
        # Load questions from JSON
        self.questions = self.load_questions()
    
    def record(self, label, confidence):
        """Record a new weighted valence score"""
        valence = self.valence_map.get(label, 0.0)
        weighted_valence = valence * confidence
        self.raw_scores.append(weighted_valence)
        
        if len(self.raw_scores) > self.max_history:
            self.raw_scores = self.raw_scores[-self.max_history:]
    
    def weighted_average(self):
        """Compute weighted average giving extra weight to last recent_fraction of frames"""
        N = len(self.raw_scores)
        if N == 0:
            return 0.0
            
        k = int(N * self.recent_fraction)
        boundary = max(0, N - k)
        sum_score = 0.0
        weight_sum = 0.0
        
        for i, score in enumerate(self.raw_scores):
            w = self.recent_weight if i >= boundary else 1.0
            sum_score += w * score
            weight_sum += w
            
        return sum_score / weight_sum if weight_sum > 0 else 0.0
    
    def average_score(self):
        """Compute simple average of all scores"""
        if not self.raw_scores:
            return 0.0
        return sum(self.raw_scores) / len(self.raw_scores)
    
    def volatility(self):
        """Compute volatility (weighted standard deviation)"""
        N = len(self.raw_scores)
        if N <= 1:
            return 0.0
            
        mu = self.weighted_average()
        k = int(N * self.recent_fraction)
        boundary = max(0, N - k)
        var_sum = 0.0
        weight_sum = 0.0
        
        for i, score in enumerate(self.raw_scores):
            w = self.recent_weight if i >= boundary else 1.0
            var_sum += w * (score - mu) ** 2
            weight_sum += w
            
        return math.sqrt(var_sum / weight_sum) if weight_sum > 0 else 0.0
    
    def reset(self):
        """Reset all scores"""
        self.raw_scores.clear()
        self.current_question_index = 0
        self.has_shown_greeting = False
    
    def load_questions(self):
        """Load questions from JSON file"""
        questions = {
            "greeting": ["Merhaba, bugün nasılsınız?"],
            "positive_high": [
                "Harika gidiyor, bir başarı hikayeniz var mı?",
                "Bu pozitif deneyimi nasıl pekiştirdiniz?"
            ],
            "positive_low": [
                "Başarılı olduğunuz bir projeyi anlatır mısınız?",
                "Bu başarıya nasıl ulaştınız?"
            ],
            "neutral": [
                "Şu ana kadar deneyiminizi genel olarak nasıl değerlendiriyorsunuz?",
                "Bu rol için sizi en çok motive eden nedir?"
            ],
            "negative_low": [
                "Zorlu bir anı paylaşır mısınız?",
                "Bu durumu nasıl aşmayı planlıyorsunuz?"
            ],
            "negative_high": [
                "Kendinizi zorlayıcı bir durumda hissettiğiniz an neydi?",
                "Bu stresi nasıl azaltabilirsiniz?"
            ],
            "unknown": [
                "Kendinizden kısaca bahseder misiniz?",
                "Bu rol sizi neden çekiyor?"
            ]
        }
        return questions
    
    def bucket_key(self, score):
        """Determine bucket key based on score"""
        if score >= 0.4:
            return "positive_high"
        elif score >= 0.1:
            return "positive_low"
        elif -0.1 < score < 0.1:
            return "neutral"
        elif score > -0.4:
            return "negative_low"
        else:
            return "negative_high"
    
    def get_current_question(self):
        """Get the current question based on state"""
        if not self.has_shown_greeting:
            return self.questions["greeting"][0]
        
        score = self.weighted_average()
        bucket = self.bucket_key(score)
        bucket_questions = self.questions.get(bucket, self.questions["unknown"])
        
        if self.current_question_index >= len(bucket_questions):
            self.current_question_index = 0
            
        return bucket_questions[self.current_question_index]
    
    def next_question(self):
        """Move to the next question"""
        if not self.has_shown_greeting:
            self.has_shown_greeting = True
            self.current_question_index = 0
        else:
            self.current_question_index += 1 