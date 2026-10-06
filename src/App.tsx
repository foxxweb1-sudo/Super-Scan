import React, { useEffect, useRef, useState } from 'react';
import { GoogleGenAI } from '@google/genai';
import { Camera, Square, Play, Loader2, AlertCircle, Activity, Cpu, ScanFace, Info, X, Sparkles, Eye } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { motion, AnimatePresence } from 'motion/react';

declare global {
  interface Window {
    aistudio?: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

interface SmoothedBox {
  x: number;
  y: number;
  width: number;
  height: number;
  class: string;
  score: number;
  opacity: number;
  labelX: number;
  labelY: number;
}

// القاموس العربي لتصنيفات الكائنات المكتشفة
const OBJECT_TRANSLATIONS: Record<string, string> = {
  person: 'شخص',
  bicycle: 'دراجة هوائية',
  car: 'سيارة',
  motorcycle: 'دراجة نارية',
  airplane: 'طائرة',
  bus: 'حافلة',
  train: 'قطار',
  truck: 'شاحنة',
  boat: 'قارب',
  'traffic light': 'إشارة مرور',
  'fire hydrant': 'صنبور إطفاء',
  'stop sign': 'علامة توقف',
  'parking meter': 'عداد وقوف',
  bench: 'مقعد',
  bird: 'طائر',
  cat: 'قطة',
  dog: 'كلب',
  horse: 'حصان',
  sheep: 'خروف',
  cow: 'بقرة',
  elephant: 'فيل',
  bear: 'دب',
  zebra: 'حمار وحشي',
  giraffe: 'زرافة',
  backpack: 'حقيبة ظهر',
  umbrella: 'مظلة',
  handbag: 'حقيبة يد',
  tie: 'ربطة عنق',
  suitcase: 'حقيبة سفر',
  frisbee: 'قرص طائر',
  skis: 'زلاجات',
  snowboard: 'لوح ثلج',
  'sports ball': 'كرة رياضية',
  kite: 'طائرة ورقية',
  'baseball bat': 'مضرب بيسبول',
  'baseball glove': 'قفاز بيسبول',
  skateboard: 'لوح تزلج',
  surfboard: 'لوح ركوب أمواج',
  'tennis racket': 'مضرب تنس',
  bottle: 'زجاجة',
  'wine glass': 'كأس',
  cup: 'كوب',
  fork: 'شوكة',
  knife: 'سكين',
  spoon: 'ملعقة',
  bowl: 'وعاء',
  banana: 'موز',
  apple: 'تفاح',
  sandwich: 'شطيرة',
  orange: 'برتقال',
  broccoli: 'بروكلي',
  carrot: 'جزر',
  'hot dog': 'هوت دوغ',
  pizza: 'بيتزا',
  donut: 'دونات',
  cake: 'كعكة',
  chair: 'كرسي',
  couch: 'أريكة',
  'potted plant': 'نبتة منزلية',
  bed: 'سرير',
  'dining table': 'طاولة طعام',
  toilet: 'مرحاض',
  tv: 'شاشة تلفاز',
  laptop: 'حاسوب محمول',
  mouse: 'فأرة حاسوب',
  remote: 'جهاز تحكم',
  keyboard: 'لوحة مفاتيح',
  'cell phone': 'هاتف ذكي',
  microwave: 'ميكروويف',
  oven: 'فرن',
  toaster: 'محمصة',
  sink: 'مغسلة',
  refrigerator: 'ثلاجة',
  book: 'كتاب',
  clock: 'ساعة',
  vase: 'مزهرية',
  scissors: 'مقص',
  'teddy bear': 'دمية دبدوب',
  'hair drier': 'مجفف شعر',
  toothbrush: 'فرشاة أسنان',
};

// القاموس العربي للمشاعر
const EMOTION_TRANSLATIONS: Record<string, string> = {
  neutral: 'طبيعي / متزن',
  happy: 'سعيد / مبتهج',
  sadness: 'حزين / تأملي',
  sad: 'حزين / تأملي',
  surprised: 'متفاجئ / مندهش',
  angry: 'غاضب / حماسي',
  fear: 'حذر / متوجس',
  disgust: 'مشمئز / منزعج',
};

function getLocalSceneDescription(objects: string[], emotion: string): string {
  if (objects.length === 0) {
    return 'مشهد هادئ، في انتظار رصد عناصر أو أشخاص في نطاق الكاميرا.';
  }
  const arabicObjs = objects.map(o => OBJECT_TRANSLATIONS[o] || o).slice(0, 3).join('، ');
  const emotionText = EMOTION_TRANSLATIONS[emotion] || emotion;
  return `بيئة تفاعلية تحتوي على: ${arabicObjs} | الحالة التعبيرية: ${emotionText}.`;
}

const getSceneAnalysisFromGemini = async (objects: string[], emotion: string): Promise<string> => {
  try {
    const apiKey = process.env.GEMINI_API_KEY || process.env.API_KEY;
    if (!apiKey || apiKey === 'dummy-key') {
      return getLocalSceneDescription(objects, emotion);
    }
    const ai = new GoogleGenAI({ apiKey });
    const prompt = `أنت نظام رؤية حاسوبية ومحلل ذكي للمشاهد. بناءً على العناصر المكتشفة: [${objects.length > 0 ? objects.join(', ') : 'لا توجد عناصر'}] وحالة الشخص: [${emotion}]، صف طبيعة المشهد والنشاط الحالي بجملة واحدة ذكية وبليغة ومختصرة باللغة العربية (أقل من 12 كلمة). لا تذكر أي أصوات أو موسيقى على الإطلاق.`;
    const response = await ai.models.generateContent({
      model: 'gemini-flash-lite-latest',
      contents: prompt,
    });
    return response.text?.trim() || getLocalSceneDescription(objects, emotion);
  } catch (e: any) {
    return getLocalSceneDescription(objects, emotion);
  }
};

export default function App() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [status, setStatus] = useState('جاري تحميل نماذج الرؤية الحاسوبية...');
  const [sceneDescription, setSceneDescription] = useState('في انتظار تشغيل الكاميرا لبدء التحليل البصري...');
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [infoMsg, setInfoMsg] = useState<string | null>(null);
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const [consoleState, setConsoleState] = useState({
    emotion: 'neutral',
    objects: [] as string[],
    blendshapes: { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 }
  });
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const faceCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  const objectModelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const isPlayingRef = useRef(false);
  const lastStateRef = useRef<string>("");
  const pendingStateRef = useRef<string | null>(null);
  const analysisTimeoutRef = useRef<any>(null);
  const lastStateUpdateTimeRef = useRef<number>(0);
  const detectLoopRef = useRef<number | null>(null);
  const smoothedBoxesRef = useRef<Map<string, SmoothedBox>>(new Map());
  const smoothedBlendshapesRef = useRef({ smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 });

  useEffect(() => {
    // تحميل نماذج TensorFlow و COCO-SSD و MediaPipe FaceLandmarker بدون أي مكتبات صوتية
    const loadModels = async () => {
      try {
        await tf.ready();
        const cocoModel = await cocoSsd.load();
        objectModelRef.current = cocoModel;

        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1
        });
        faceLandmarkerRef.current = faceLandmarker;

        setIsModelLoaded(true);
        setStatus('في وضع الاستعداد');
      } catch (err: any) {
        console.error("Failed to load models:", err);
        setStatus('خطأ في تحميل النماذج');
        setErrorMsg(err.message || 'تعذر تحميل نماذج الرؤية الحاسوبية');
      }
    };
    
    loadModels();
    
    return () => {
      stopSession();
    };
  }, []);

  const runDetection = async () => {
    if (!isPlayingRef.current || !videoRef.current || !canvasRef.current || !objectModelRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (video.readyState >= 2 && ctx) {
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      try {
        const predictions = await objectModelRef.current.detect(video);
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const detectedClasses = new Set<string>();

        // منطق تنعيم المربعات وتتبعها
        const newSmoothedBoxes = new Map<string, SmoothedBox>();
        const unassignedPredictions = [...predictions];

        smoothedBoxesRef.current.forEach((box, id) => {
          let closestIdx = -1;
          let minDist = Infinity;
          unassignedPredictions.forEach((pred, idx) => {
            if (pred.class === box.class) {
              const [px, py, pw, ph] = pred.bbox;
              const dist = Math.hypot(px + pw/2 - (box.x + box.width/2), py + ph/2 - (box.y + box.height/2));
              if (dist < 150) {
                if (dist < minDist) {
                  minDist = dist;
                  closestIdx = idx;
                }
              }
            }
          });

          if (closestIdx !== -1) {
            const pred = unassignedPredictions[closestIdx];
            const [px, py, pw, ph] = pred.bbox;
            const lerp = 0.15;
            box.x += (px - box.x) * lerp;
            box.y += (py - box.y) * lerp;
            box.width += (pw - box.width) * lerp;
            box.height += (ph - box.height) * lerp;
            box.opacity = Math.min(1, box.opacity + 0.1);
            box.score = pred.score;
            
            const targetLabelX = box.x + box.width + 20;
            const targetLabelY = box.y - 20;
            box.labelX += (targetLabelX - box.labelX) * lerp;
            box.labelY += (targetLabelY - box.labelY) * lerp;

            newSmoothedBoxes.set(id, box);
            unassignedPredictions.splice(closestIdx, 1);
            detectedClasses.add(box.class);
          } else {
            box.opacity -= 0.05;
            if (box.opacity > 0) {
              newSmoothedBoxes.set(id, box);
              detectedClasses.add(box.class);
            }
          }
        });

        unassignedPredictions.forEach((pred) => {
          const id = Math.random().toString(36).substring(7);
          const [x, y, width, height] = pred.bbox;
          newSmoothedBoxes.set(id, {
            x, y, width, height, class: pred.class, score: pred.score, opacity: 0,
            labelX: x + width + 40, labelY: y - 40
          });
          detectedClasses.add(pred.class);
        });

        smoothedBoxesRef.current = newSmoothedBoxes;

        // رسم مربعات الاستهداف والتعرف على العناصر
        smoothedBoxesRef.current.forEach((box) => {
          const { x, y, width, height, opacity, labelX, labelY } = box;
          const arabicName = OBJECT_TRANSLATIONS[box.class] || box.class;
          const text = `${arabicName} (${Math.round(box.score * 100)}%)`;

          ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.8})`;
          ctx.lineWidth = 1;

          // زوايا الاستهداف (HUD Corners)
          const cornerLength = Math.min(15, width / 4, height / 4);
          ctx.beginPath();
          ctx.moveTo(x, y + cornerLength);
          ctx.lineTo(x, y);
          ctx.lineTo(x + cornerLength, y);
          
          ctx.moveTo(x + width - cornerLength, y);
          ctx.lineTo(x + width, y);
          ctx.lineTo(x + width, y + cornerLength);
          
          ctx.moveTo(x + width, y + height - cornerLength);
          ctx.lineTo(x + width, y + height);
          ctx.lineTo(x + width - cornerLength, y + height);
          
          ctx.moveTo(x + cornerLength, y + height);
          ctx.lineTo(x, y + height);
          ctx.lineTo(x, y + height - cornerLength);
          ctx.stroke();

          // تقاطع المنتصف (Crosshair)
          ctx.beginPath();
          ctx.moveTo(x + width / 2 - 5, y + height / 2);
          ctx.lineTo(x + width / 2 + 5, y + height / 2);
          ctx.moveTo(x + width / 2, y + height / 2 - 5);
          ctx.lineTo(x + width / 2, y + height / 2 + 5);
          ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.4})`;
          ctx.stroke();

          // خط مؤشر التسمية
          ctx.beginPath();
          ctx.moveTo(x + width, y);
          ctx.lineTo(labelX, labelY + 16);
          ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.5})`;
          ctx.setLineDash([2, 2]);
          ctx.stroke();
          ctx.setLineDash([]);

          // شريط التسمية بالعربية
          ctx.font = '600 11px "Cairo", "JetBrains Mono", sans-serif';
          const textWidth = ctx.measureText(text).width;
          ctx.fillStyle = `rgba(0, 0, 0, ${opacity * 0.7})`;
          ctx.fillRect(labelX - 4, labelY - 2, textWidth + 14, 20);
          ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.4})`;
          ctx.strokeRect(labelX - 4, labelY - 2, textWidth + 14, 20);
          ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
          ctx.fillText(text, labelX + 3, labelY + 13);
        });

        const classesArray = Array.from(detectedClasses).sort();
        
        let currentEmotion = "neutral";
        let currentBlendshapes = { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 };
        
        if (faceLandmarkerRef.current) {
          const faceResult = faceLandmarkerRef.current.detectForVideo(video, performance.now());
          
          // رسم شبكة الوجه النقطية على الكانفاس الثانوي (Biometric Scan Canvas)
          if (faceCanvasRef.current) {
            const fCanvas = faceCanvasRef.current;
            const fCtx = fCanvas.getContext('2d');
            if (fCtx) {
              fCtx.clearRect(0, 0, fCanvas.width, fCanvas.height);
              
              if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
                const time = performance.now() / 1500;
                
                let minX = video.videoWidth, maxX = 0, minY = video.videoHeight, maxY = 0;
                for (const pt of faceResult.faceLandmarks[0]) {
                  const px = pt.x * video.videoWidth;
                  const py = pt.y * video.videoHeight;
                  if (px < minX) minX = px;
                  if (px > maxX) maxX = px;
                  if (py < minY) minY = py;
                  if (py > maxY) maxY = py;
                }
                const faceWidth = maxX - minX;
                const faceHeight = maxY - minY;
                const centerX = minX + faceWidth / 2;
                const centerY = minY + faceHeight / 2;
                
                const scanY = minY + ((Math.sin(time) + 1) / 2) * faceHeight;
                const scale = Math.min(fCanvas.width / faceWidth, fCanvas.height / faceHeight) * 0.8;

                for (const pt of faceResult.faceLandmarks[0]) {
                  const px = pt.x * video.videoWidth;
                  const py = pt.y * video.videoHeight;
                  
                  const dist = Math.abs(py - scanY) / faceHeight;
                  const opacity = Math.max(0.15, 1.0 - dist * 4); 
                  
                  fCtx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
                  fCtx.beginPath();
                  
                  const drawX = fCanvas.width/2 + (px - centerX) * scale;
                  const drawY = fCanvas.height/2 + (py - centerY) * scale;
                  
                  fCtx.arc(drawX, drawY, 1.5, 0, 2 * Math.PI);
                  fCtx.fill();
                }
              }
            }
          }

          if (faceResult.faceBlendshapes && faceResult.faceBlendshapes.length > 0) {
            const blendshapes = faceResult.faceBlendshapes[0].categories;
            const getScore = (name: string) => blendshapes.find(b => b.categoryName === name)?.score || 0;
            
            // مؤشرات تعابير الوجه
            currentBlendshapes.smile = (getScore('mouthSmileLeft') + getScore('mouthSmileRight')) / 2;
            currentBlendshapes.frown = Math.min(1, (getScore('mouthFrownLeft') + getScore('mouthFrownRight') + getScore('mouthRollLower')) * 5);
            currentBlendshapes.mouthOpen = getScore('jawOpen');
            currentBlendshapes.browRaise = (getScore('browInnerUp') + getScore('browOuterUpLeft') + getScore('browOuterUpRight')) / 3;
            currentBlendshapes.eyeBlink = (getScore('eyeBlinkLeft') + getScore('eyeBlinkRight')) / 2;
            currentBlendshapes.pucker = getScore('mouthPucker');
            
            const surpriseScore = (getScore('jawOpen') + getScore('browInnerUp')) / 2;
            const angerScore = (getScore('browDownLeft') + getScore('browDownRight') + getScore('mouthPressLeft')) / 3;
            const fearScore = ((getScore('jawOpen') + getScore('browInnerUp') + getScore('mouthStretchLeft') + getScore('mouthStretchRight')) / 4) * 0.6;
            const disgustScore = Math.min(1, (getScore('noseSneerLeft') + getScore('noseSneerRight') + getScore('mouthUpperUpLeft') + getScore('mouthUpperUpRight')) * 4);
            
            const emotions = [
              { name: 'happy', score: currentBlendshapes.smile },
              { name: 'sadness', score: currentBlendshapes.frown },
              { name: 'surprised', score: surpriseScore },
              { name: 'angry', score: angerScore },
              { name: 'fear', score: fearScore },
              { name: 'disgust', score: disgustScore }
            ];
            
            const maxEmotion = emotions.reduce((max, e) => e.score > max.score ? e : max, emotions[0]);
            if (maxEmotion.score > 0.2) {
              currentEmotion = maxEmotion.name;
            } else {
              currentEmotion = "neutral";
            }
          }

          // رسم المسح الضوئي والإبرازات الحيوية على الكانفاس الرئيسي
          if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
            const landmarks = faceResult.faceLandmarks[0];
            const scanTime = performance.now() / 40000;
            const scanPhase = scanTime % 1;
            
            let minX = 1, maxX = 0, minY = 1, maxY = 0;
            for (const pt of landmarks) {
              if (pt.x < minX) minX = pt.x;
              if (pt.x > maxX) maxX = pt.x;
              if (pt.y < minY) minY = pt.y;
              if (pt.y > maxY) maxY = pt.y;
            }
            
            const scanProgress = (Math.sin(scanPhase * Math.PI * 2) + 1) / 2;
            const scanY = minY + scanProgress * (maxY - minY);
            const lineOpacity = Math.sin(scanProgress * Math.PI) * 0.3;
            
            ctx.save();
            
            const faceOvalIndices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
            ctx.beginPath();
            for (let i = 0; i < faceOvalIndices.length; i++) {
              const pt = landmarks[faceOvalIndices[i]];
              if (i === 0) ctx.moveTo(pt.x * canvas.width, pt.y * canvas.height);
              else ctx.lineTo(pt.x * canvas.width, pt.y * canvas.height);
            }
            ctx.closePath();
            ctx.clip();

            if (lineOpacity > 0.01) {
              ctx.fillStyle = '#ffffff';
              ctx.shadowColor = '#ffffff';
              ctx.shadowBlur = 10;
              
              for (const pt of landmarks) {
                const dist = Math.abs(pt.y - scanY);
                const threshold = 0.04;
                
                if (dist < threshold) {
                  const ptOpacity = (1 - (dist / threshold)) * lineOpacity * 2;
                  ctx.globalAlpha = Math.min(ptOpacity, 1);
                  
                  ctx.beginPath();
                  ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 1.5, 0, Math.PI * 2);
                  ctx.fill();
                }
              }
              ctx.globalAlpha = 1.0;
            }
            
            ctx.restore();

            const drawFeatureHighlight = (x: number, y: number, label: string, intensity: number) => {
              if (isNaN(intensity) || intensity < 0.05) return;
              ctx.save();
              ctx.translate(x * canvas.width, y * canvas.height);
              
              const size = 5 + intensity * 15;
              
              ctx.strokeStyle = `rgba(255, 255, 255, ${intensity * 0.8})`;
              ctx.lineWidth = 1.5;
              
              ctx.beginPath();
              ctx.moveTo(-size, -size/2);
              ctx.lineTo(-size, -size);
              ctx.lineTo(-size/2, -size);
              
              ctx.moveTo(size, -size/2);
              ctx.lineTo(size, -size);
              ctx.lineTo(size/2, -size);
              
              ctx.moveTo(-size, size/2);
              ctx.lineTo(-size, size);
              ctx.lineTo(-size/2, size);
              
              ctx.moveTo(size, size/2);
              ctx.lineTo(size, size);
              ctx.lineTo(size/2, size);
              ctx.stroke();
              
              ctx.fillStyle = `rgba(255, 255, 255, ${intensity})`;
              ctx.beginPath();
              ctx.arc(0, 0, 2, 0, Math.PI * 2);
              ctx.fill();
              
              ctx.fillStyle = `rgba(255, 255, 255, ${intensity * 0.9})`;
              ctx.font = '600 9px "Cairo", monospace';
              ctx.fillText(label, size + 5, 3);
              
              ctx.restore();
            };

            const smoothed = smoothedBlendshapesRef.current;
            drawFeatureHighlight(landmarks[61].x, landmarks[61].y, 'ابتسامة_يسار', smoothed.smile);
            drawFeatureHighlight(landmarks[291].x, landmarks[291].y, 'ابتسامة_يمين', smoothed.smile);
            
            drawFeatureHighlight(landmarks[61].x, landmarks[61].y, 'عبوس_يسار', smoothed.frown);
            drawFeatureHighlight(landmarks[291].x, landmarks[291].y, 'عبوس_يمين', smoothed.frown);
            
            drawFeatureHighlight(landmarks[52].x, landmarks[52].y, 'حاجب_يسار', smoothed.browRaise);
            drawFeatureHighlight(landmarks[282].x, landmarks[282].y, 'حاجب_يمين', smoothed.browRaise);
            
            drawFeatureHighlight(landmarks[152].x, landmarks[152].y, 'فتح_الفك', smoothed.mouthOpen);
            drawFeatureHighlight(landmarks[13].x, landmarks[13].y, 'زم_الشفاه', smoothed.pucker);
            
            drawFeatureHighlight(landmarks[159].x, landmarks[159].y, 'طرف_يسار', smoothed.eyeBlink);
            drawFeatureHighlight(landmarks[386].x, landmarks[386].y, 'طرف_يمين', smoothed.eyeBlink);
          }
        }

        const now = performance.now();
        if (now - lastStateUpdateTimeRef.current > 100) {
          const smoothingFactor = 0.15;
          const smoothed = smoothedBlendshapesRef.current;
          smoothed.smile += (currentBlendshapes.smile - smoothed.smile) * smoothingFactor;
          smoothed.frown += (currentBlendshapes.frown - smoothed.frown) * smoothingFactor;
          smoothed.mouthOpen += (currentBlendshapes.mouthOpen - smoothed.mouthOpen) * smoothingFactor;
          smoothed.browRaise += (currentBlendshapes.browRaise - smoothed.browRaise) * smoothingFactor;
          smoothed.eyeBlink += (currentBlendshapes.eyeBlink - smoothed.eyeBlink) * smoothingFactor;
          smoothed.pucker += (currentBlendshapes.pucker - smoothed.pucker) * smoothingFactor;

          setConsoleState({
            emotion: currentEmotion,
            objects: classesArray,
            blendshapes: { ...smoothed }
          });
          lastStateUpdateTimeRef.current = now;
        }

        const stateString = `${classesArray.join(',')}|${currentEmotion}`;
        
        if (stateString !== pendingStateRef.current) {
          pendingStateRef.current = stateString;
          
          if (analysisTimeoutRef.current) {
            clearTimeout(analysisTimeoutRef.current);
          }
          
          analysisTimeoutRef.current = setTimeout(async () => {
            if (stateString !== lastStateRef.current) {
              lastStateRef.current = stateString;
              const newDescription = await getSceneAnalysisFromGemini(classesArray, currentEmotion);
              setSceneDescription(newDescription);
            }
          }, 2500);
        }
      } catch (err) {
        console.error("Detection error:", err);
      }
    }
    
    if (isPlayingRef.current) {
      detectLoopRef.current = requestAnimationFrame(runDetection);
    }
  };

  const startSession = async () => {
    if (!isModelLoaded) return;
    
    try {
      setErrorMsg(null);
      setStatus('جاري تشغيل الكاميرا...');
      
      let stream = streamRef.current;
      if (!stream) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
              width: { ideal: 1280 }, 
              height: { ideal: 720 },
              facingMode: 'user'
            } 
          });
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play().catch(e => console.error("Video play error:", e));
          }
          setIsCameraActive(true);
        } catch (camErr: any) {
          console.error("Camera error:", camErr);
          setStatus('خطأ في الكاميرا');
          setErrorMsg('تم رفض إذن الوصول للكاميرا. يرجى تفعيل إذن الكاميرا في إعدادات المتصفح ثم إعادة تحميل الصفحة.');
          return;
        }
      }

      if (!isPlayingRef.current) {
        isPlayingRef.current = true;
        detectLoopRef.current = requestAnimationFrame(runDetection);
      }

      setStatus('نظام المسح الذكي نشط ويعمل الآن');
      setIsPlaying(true);
      setSceneDescription('جاري رصد وتحليل المشهد والعناصر المحيطة...');

    } catch (err: any) {
      console.error("Setup Error:", err);
      setStatus('فشل تشغيل النظام');
      setErrorMsg(err.message || 'حدث خطأ غير متوقع أثناء تشغيل النظام.');
      setInfoMsg(null);
      stopSession(false);
    }
  };

  const stopSession = (closeCamera: boolean = true) => {
    setIsPlaying(false);
    if (analysisTimeoutRef.current) {
      clearTimeout(analysisTimeoutRef.current);
      analysisTimeoutRef.current = null;
    }
    pendingStateRef.current = null;
    
    setStatus('في وضع الاستعداد');
    
    setConsoleState({
      emotion: 'neutral',
      objects: [],
      blendshapes: { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 }
    });
    smoothedBlendshapesRef.current = { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 };
    smoothedBoxesRef.current.clear();
    
    setInfoMsg(null);

    if (closeCamera) {
      isPlayingRef.current = false;
      setSceneDescription('في انتظار تشغيل الكاميرا لبدء التحليل البصري...');
      
      if (detectLoopRef.current) {
        cancelAnimationFrame(detectLoopRef.current);
        detectLoopRef.current = null;
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        setIsCameraActive(false);
      }
    }
  };

  return (
    <div dir="rtl" className="h-[100dvh] w-full bg-black text-white flex overflow-hidden font-sans relative selection:bg-white selection:text-black">
      {/* Background Camera Feed */}
      <div className="absolute inset-0 z-0">
        {!isCameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white/50 z-10 font-mono text-sm gap-2">
            <Camera className="w-10 h-10 mb-2 opacity-50 stroke-[1.5]" />
            <p className="tracking-widest uppercase text-xs font-semibold text-white/70">SUPER_SCAN.CAMERA_OFFLINE</p>
            <p className="text-xs text-white/40 font-sans">اضغط على زر "بدء تشغيل النظام" لتفعيل المسح الذكي الصامت</p>
          </div>
        )}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`absolute inset-0 w-full h-full object-cover grayscale contrast-125 opacity-60 transition-opacity duration-500 ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
        />
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 w-full h-full object-cover pointer-events-none transition-opacity duration-500 z-[15] ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
        />
        {/* Vignette & Scanlines */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.85)_100%)] z-10" />
        <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.25)_50%)] bg-[length:100%_4px] z-10" />
        
        {/* Decorative HUD Elements */}
        <div className="absolute inset-0 pointer-events-none z-10 flex items-center justify-center overflow-hidden">
          <div className="w-[150vw] h-[150vw] sm:w-[600px] sm:h-[600px] border border-white/10 rounded-full border-dashed animate-[spin_60s_linear_infinite] shrink-0" />
          <div className="absolute w-[100vw] h-[100vw] sm:w-[400px] sm:h-[400px] border border-white/5 rounded-full animate-[spin_40s_linear_infinite_reverse] shrink-0" />
          <div className="absolute w-px h-full bg-white/5" />
          <div className="absolute h-px w-full bg-white/5" />
        </div>
      </div>

      {/* Main HUD Interface */}
      <div className="relative z-20 w-full h-full pointer-events-auto p-4 sm:p-6 overflow-y-auto overflow-x-hidden pb-32 sm:pb-6">
        <div className="flex flex-col lg:flex-row justify-between gap-4 min-h-full">
          
          {/* Right Column in RTL (Top/Start: System Status & Mobile Controls) */}
          <div className="contents lg:flex lg:flex-col lg:justify-between w-full lg:w-84 pointer-events-none shrink-0">
            
            {/* Top Bar: System Status */}
            <div className="flex flex-col gap-4 shrink-0 order-1 lg:order-none pointer-events-auto">
              
              <div className="flex flex-col items-start gap-3 shrink-0">
                <div className="flex items-start justify-between w-full">
                  <div>
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-white animate-pulse" />
                      <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white drop-shadow-[0_0_12px_rgba(255,255,255,0.7)]">
                        SUPER SCAN
                      </h1>
                    </div>
                    <p className="text-[11px] text-white/80 font-mono tracking-wider mt-0.5">
                      سوبر سكان • محرك الرؤية الحاسوبية والمسح الذكي v2.4
                    </p>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {/* زر تشغيل مخصص للهواتف في الوضع الأفقي */}
                    <button
                      onClick={() => isCameraActive ? stopSession(true) : startSession()}
                      disabled={!isModelLoaded}
                      className={`hidden landscape:flex lg:landscape:hidden justify-center items-center gap-2 px-3 py-2 text-[11px] font-bold transition-all duration-300 border backdrop-blur-md cursor-pointer ${
                        isCameraActive 
                          ? 'bg-red-500/20 text-red-400 border-red-500 hover:bg-red-500/30 shadow-[0_0_10px_rgba(239,68,68,0.4)]' 
                          : 'bg-white/10 text-white border-white hover:bg-white/20 shadow-[0_0_10px_rgba(255,255,255,0.3)]'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {!isModelLoaded ? (
                        <><Loader2 className="w-3 h-3 animate-spin" /> جاري التهيئة</>
                      ) : isCameraActive ? (
                        <><Square className="w-3 h-3 fill-current" /> إيقاف</>
                      ) : (
                        <><Play className="w-3 h-3 fill-current" /> تشغيل</>
                      )}
                    </button>
                    
                    {/* زر المعلومات */}
                    <button 
                      onClick={() => setIsInfoOpen(true)}
                      className="p-2.5 bg-white/10 hover:bg-white/20 rounded-full transition-colors backdrop-blur-md border border-white/20 shrink-0 text-white cursor-pointer"
                      title="معلومات عن سوبر سكان"
                    >
                      <Info className="w-5 h-5 text-white" />
                    </button>
                  </div>
                </div>

                {/* مؤشر حالة النظام */}
                <div className="text-xs font-mono text-white/90 flex items-center gap-2.5 bg-black/50 backdrop-blur-md px-3.5 py-2 border border-white/20 rounded-none shadow-sm">
                  <div className={`w-2.5 h-2.5 rounded-full shrink-0 ${
                    status.includes('نشط') || isPlaying 
                      ? 'bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.9)] animate-pulse' 
                      : status.includes('جاري') || status.includes('تحميل') 
                      ? 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.8)]' 
                      : status.includes('خطأ') || status.includes('فشل') 
                      ? 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]' 
                      : 'bg-zinc-500'
                  }`} />
                  <span className="font-sans font-medium text-xs">{status}</span>
                </div>
              </div>

              {/* زر تشغيل الموبايل */}
              <div className="flex lg:hidden landscape:hidden flex-col items-stretch gap-4 shrink-0">
                <button
                  onClick={() => isCameraActive ? stopSession(true) : startSession()}
                  disabled={!isModelLoaded}
                  className={`flex justify-center items-center gap-3 px-8 py-3.5 text-sm font-bold tracking-wide transition-all duration-300 border-2 backdrop-blur-md cursor-pointer ${
                    isCameraActive 
                      ? 'bg-red-500/20 text-red-300 border-red-500 hover:bg-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.4)]' 
                      : 'bg-white/10 text-white border-white hover:bg-white/20 shadow-[0_0_20px_rgba(255,255,255,0.3)]'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  {!isModelLoaded ? (
                    <><Loader2 className="w-5 h-5 animate-spin" /> جاري التهيئة...</>
                  ) : isCameraActive ? (
                    <><Square className="w-5 h-5 fill-current" /> إيقاف النظام</>
                  ) : (
                    <><Play className="w-5 h-5 fill-current" /> بدء تشغيل النظام</>
                  )}
                </button>
              </div>

              {/* مساحة المعاينة على الموبايل */}
              <div className="h-[45vh] landscape:h-[100vh] lg:hidden pointer-events-none shrink-0" />
            </div>

            {/* بطاقات المسح الحيوي والحالة الشعورية */}
            <div className="flex flex-col landscape:flex-row lg:landscape:flex-col gap-4 shrink-0 lg:mt-auto order-4 lg:order-none pointer-events-auto">
              
              {/* ماسح الوجه النقطي */}
              <div className="flex flex-col justify-center shrink-0 landscape:flex-1 lg:landscape:flex-none">
                <div className="bg-black/50 backdrop-blur-md border border-white/20 p-4 w-full shadow-[0_0_30px_rgba(0,0,0,0.8)] relative overflow-hidden flex flex-col h-64 landscape:h-full lg:landscape:h-64 shrink-0" title="المسح الحيوي لمعالم الوجه في الوقت الفعلي">
                  <h3 className="text-xs font-bold text-white/70 tracking-wider mb-2.5 shrink-0 flex items-center gap-2">
                    <ScanFace className="w-4 h-4 text-white/80" />
                    المسح الحيوي (Biometric Scan)
                  </h3>
                  <div className="relative w-full flex-1 border border-white/10 flex items-center justify-center bg-white/5 min-h-0">
                    <canvas
                      ref={faceCanvasRef}
                      width={300}
                      height={300}
                      className={`w-full h-full object-contain transition-opacity duration-500 ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
                    />
                  </div>
                </div>
              </div>

              {/* بطاقة الحالة التعبيرية والمشاعر */}
              <div className="flex flex-col justify-end shrink-0 landscape:flex-1 lg:landscape:flex-none">
                <div className="bg-black/50 backdrop-blur-md border border-white/20 p-5 w-full h-full shadow-[0_0_30px_rgba(0,0,0,0.8)]" title="تحليل الحالة الشعورية والمزاجية عبر تعابير الوجه">
                  <h3 className="text-xs font-bold text-white/70 tracking-wider mb-2.5 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-white/80" />
                    الحالة التعبيرية والشعورية
                  </h3>
                  <div className="text-2xl font-bold tracking-tight mb-4 text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.5)]">
                    {EMOTION_TRANSLATIONS[consoleState.emotion] || consoleState.emotion}
                  </div>
                  
                  <div className="space-y-2.5">
                    {[
                      { label: 'الابتسامة', value: consoleState.blendshapes.smile },
                      { label: 'العبوس', value: consoleState.blendshapes.frown },
                      { label: 'فتح الفم', value: consoleState.blendshapes.mouthOpen },
                      { label: 'رفع الحاجبين', value: consoleState.blendshapes.browRaise },
                      { label: 'طرف العين', value: consoleState.blendshapes.eyeBlink },
                      { label: 'زم الشفاه', value: consoleState.blendshapes.pucker },
                    ].map((item) => (
                      <div key={item.label}>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-white/70 font-medium">{item.label}</span>
                          <span className="font-mono font-bold text-white/90">
                            {isNaN(item.value) ? 0 : (item.value * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="h-1 bg-white/10 overflow-hidden rounded-full">
                          <motion.div 
                            className="h-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.9)] rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: `${isNaN(item.value) ? 0 : item.value * 100}%` }}
                            transition={{ type: 'spring', bounce: 0, duration: 0.5 }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

            </div>
          </div>

          {/* Left Column in RTL (Top/End: Main Desktop Button, Entities & Scene Analysis) */}
          <div className="contents lg:flex lg:flex-col lg:justify-between lg:items-start w-full lg:w-84 pointer-events-none shrink-0 mt-0">
            
            {/* Top Desktop Controls */}
            <div className="hidden lg:flex flex-col items-start gap-4 shrink-0 order-none pointer-events-auto">
              <button
                onClick={() => isCameraActive ? stopSession(true) : startSession()}
                disabled={!isModelLoaded}
                className={`flex justify-center items-center gap-3 px-10 py-4 text-sm font-bold tracking-wider transition-all duration-300 border-2 backdrop-blur-md cursor-pointer ${
                  isCameraActive 
                    ? 'bg-red-500/20 text-red-300 border-red-500 hover:bg-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.4)]' 
                    : 'bg-white/10 text-white border-white hover:bg-white/20 shadow-[0_0_20px_rgba(255,255,255,0.3)]'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {!isModelLoaded ? (
                  <><Loader2 className="w-5 h-5 animate-spin" /> جاري التهيئة...</>
                ) : isCameraActive ? (
                  <><Square className="w-5 h-5 fill-current" /> إيقاف النظام</>
                ) : (
                  <><Play className="w-5 h-5 fill-current" /> بدء تشغيل النظام</>
                )}
              </button>
            </div>

            {/* Bottom: Entities & Scene Analysis */}
            <div className="flex flex-col gap-4 shrink-0 w-full order-2 lg:order-none pointer-events-auto">
              
              {/* بطاقة العناصر المكتشفة */}
              <div className="w-full bg-black/50 backdrop-blur-md border border-white/20 p-5 shadow-[0_0_30px_rgba(0,0,0,0.8)]" title="الكائنات المكتشفة في مشهد الكاميرا">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xs font-bold text-white/70 tracking-wider flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-white/80" />
                    العناصر المكتشفة (Entities)
                  </h3>
                  <span className="text-[10px] font-mono px-2 py-0.5 bg-white/10 text-white/80 border border-white/20">
                    {consoleState.objects.length}
                  </span>
                </div>
                {consoleState.objects.length === 0 ? (
                  <p className="text-xs text-white/40 italic">لم يتم رصد عناصر محددة حالياً.</p>
                ) : (
                  <ul className="space-y-2">
                    <AnimatePresence>
                      {consoleState.objects.map((obj) => (
                        <motion.li 
                          key={obj}
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -10 }}
                          className="text-xs flex items-center justify-between text-white/90 py-1 border-b border-white/5 last:border-0"
                        >
                          <div className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 bg-white shadow-[0_0_6px_rgba(255,255,255,0.9)]" />
                            <span className="font-semibold">{OBJECT_TRANSLATIONS[obj] || obj}</span>
                          </div>
                          <span className="text-[10px] font-mono text-white/50 uppercase tracking-widest">{obj}</span>
                        </motion.li>
                      ))}
                    </AnimatePresence>
                  </ul>
                )}
              </div>

              {/* بطاقة تحليل المشهد والبيئة الذكية */}
              <div className="w-full bg-black/50 backdrop-blur-md border border-white/20 p-5 shadow-[0_0_30px_rgba(0,0,0,0.8)]" title="تحليل بصري ذكي لطبيعة المشهد والنشاط">
                <h3 className="text-xs font-bold text-white/70 tracking-wider mb-3 flex items-center gap-2">
                  <Eye className="w-4 h-4 text-white/80" />
                  تحليل المشهد والبيئة الذكية
                </h3>
                <div className="relative overflow-hidden pr-3 border-r-2 border-white">
                  <p className="text-xs leading-relaxed text-white/95 font-medium">
                    {sceneDescription}
                  </p>
                </div>
              </div>

            </div>
          </div>

        </div>
      </div>

      {/* نافذة الخطأ (Error Modal) */}
      <AnimatePresence>
        {errorMsg && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto"
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-zinc-900 border border-red-500/50 p-6 max-w-md w-full shadow-[0_0_40px_rgba(239,68,68,0.25)] relative text-right"
            >
              <div className="flex items-start gap-4 mb-6">
                <div className="p-3 bg-red-500/10 border border-red-500/30 shrink-0">
                  <AlertCircle className="w-6 h-6 text-red-500" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-red-400">{status}</h3>
                  <p className="text-xs mt-2 text-red-300/80 leading-relaxed">{errorMsg}</p>
                </div>
              </div>
              
              <button 
                onClick={() => setErrorMsg(null)}
                className={`w-full py-3 text-xs font-bold transition-colors cursor-pointer ${
                  status.includes('الكاميرا') 
                    ? 'bg-red-500/20 hover:bg-red-500/30 border border-red-500/50 text-red-300' 
                    : 'bg-white/10 hover:bg-white/20 border border-white/20 text-white'
                }`}
              >
                إغلاق وتجاهل
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* نافذة "عن سوبر سكان" (About Modal) */}
      <AnimatePresence>
        {isInfoOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto"
            onClick={() => setIsInfoOpen(false)}
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-zinc-900 border border-white/20 p-6 max-w-lg w-full shadow-[0_0_40px_rgba(0,0,0,0.8)] relative max-h-[90vh] overflow-y-auto text-right"
            >
              <div className="flex items-start justify-between gap-4 mb-5 border-b border-white/10 pb-4">
                <div className="flex items-center gap-2.5">
                  <Sparkles className="w-5 h-5 text-white" />
                  <h2 className="text-lg font-bold text-white">
                    عن سوبر سكان (Super Scan)
                  </h2>
                </div>
                <button 
                  onClick={() => setIsInfoOpen(false)}
                  className="p-1.5 border border-white/20 bg-black/50 hover:bg-white/10 text-white/60 hover:text-white transition-colors cursor-pointer"
                  title="إغلاق"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              <div className="space-y-4 text-xs sm:text-sm text-white/85 leading-relaxed">
                <p>
                  <strong>سوبر سكان (Super Scan)</strong> هو نظام ذكاء اصطناعي متطور للمسح البصري، يدمج بين الرؤية الحاسوبية اللحظية وتحليل المشاعر وتتبع الكائنات عبر كاميرا جهازك في بيئة هادئة بدون أي تشويش صوتي.
                </p>
                <p>
                  يقوم النظام بتحليل تعابير وجهك وتحديد العناصر الموجودة حولك في الوقت الفعلي لتوفير قراءة ذكية لحالة المشهد وبيئته.
                </p>
                <ul className="list-disc pr-5 space-y-2.5 text-white/75 text-xs">
                  <li>
                    <strong className="text-white">المسح الحيوي (Biometric Scan):</strong> يتتبع معالم الوجه (Face Mesh) ونقاط التعابير لاستنتاج حالتك المزاجية (فرح، حزن، مفاجأة، حماس، هدوء...).
                  </li>
                  <li>
                    <strong className="text-white">التعرف على الكائنات (Entities):</strong> يكتشف الأشياء في محيطك (مثل الهواتف، الحواسيب، الأكواب، الكتب) لحظياً.
                  </li>
                  <li>
                    <strong className="text-white">تحليل المشهد الذكي:</strong> يبتكر الذكاء الاصطناعي وصفاً دقيقاً لطبيعة البيئة الحالية ومجال التركيز.
                  </li>
                </ul>
                <div className="p-3 bg-white/5 border border-white/10 text-[11px] text-white/60 mt-4 leading-normal">
                  <strong>ملاحظة الخصوصية:</strong> تتم جميع عمليات المعالجة البصرية محلياً داخل متصفحك أو عبر قنوات آمنة ومحمية، ولا يتم حفظ أو تسجيل أي لقطات فيديو على الإطلاق.
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
