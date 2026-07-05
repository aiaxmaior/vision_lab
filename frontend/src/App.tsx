/**
 * 🎭 VISION LAB v4.0 - React Frontend
 * 
 * Title:      Vision Lab Main Application
 * Author:     ajax
 * Date:       2026-03-09
 * Version:    4.0.0
 * License:    MIT
 * 
 * Description:
 *   Dual-pane VLM interface with persistent chat, batch captioning,
 *   screenshot capture for VLM reasoning review, and Qwen3.5 thinking
 *   mode support. Left pane: functions. Right pane: always-visible chat.
 * 
 * Tech Stack:
 *   - React 18 with TypeScript
 *   - Vite for bundling
 *   - Lucide React for icons
 *   - React Markdown with syntax highlighting
 *   - html2canvas for screenshot capture
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  Upload, Settings, Cpu, MessageSquare, Send, Trash2,
  ChevronDown, ChevronRight, RefreshCw, Paperclip, X,
  Scissors, Check, Zap, Eye, Brain, Film, Camera, Monitor,
  FolderOpen, Save, ChevronLeft, FileText, ImageIcon, Download, RotateCcw, RotateCw, EyeOff,
  Volume2, VolumeX, Square
} from 'lucide-react';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  hasMedia?: boolean;
  mediaId?: string;            // uploaded media id (backend/uploads) attached to this turn, for lazy re-captioning
  mediaType?: string;          // 'image' | 'video' of the attached upload
  screenshot?: string;
  observation?: string;        // pass-A observation for media-attached chat turns
  mediaCaption?: string;       // durable text caption, generated lazily when media falls out of the image window
  autonomous?: boolean;        // assistant turn the live loop spoke unprompted (not a reply to a user message)
}

interface MediaInfo {
  media_type?: string;
  width?: number;
  height?: number;
  res?: string;
  dur?: number;
  fps?: number;
  frames?: number;
  video_codec?: string;
  audio_codec?: string;
  bitrate?: number;
}

interface UploadedMedia {
  id: string;
  filename: string;
  path: string;
  info: MediaInfo;
  thumbnail?: string;
}

interface CaptionPair {
  image_path: string;
  caption_path: string;
  filename: string;
  caption: string;
  has_caption: boolean;
}

interface DualCaptionPair {
  image_path: string;
  filename: string;
  caption_path_a: string;
  caption_a: string;
  has_caption_a: boolean;
  caption_path_b: string;
  caption_b: string;
  has_caption_b: boolean;
}

interface BatchSubdir {
  dir: string;
  rel_dir: string;
  images: CaptionPair[];
  total: number;
}

interface TokenEstimate {
  status: string;
  visual_tokens?: number;
  remaining?: number;
  context_limit?: number;
  limit_source?: string;
  media_type?: string;
  message?: string;
}

interface Config {
  api_url: string;
  model_name: string;
  processing_mode: string;
  sampling_mode: string;
  interval: number;
  target_fps: number;
  max_frames_limit: number;
  resolution_mode: string;
  image_width: number;
  image_height: number;
  system_prompt: string;
  interaction_mode: string;
  active_character: string;
  custom_mode: boolean;
  inject_thinking_tags: boolean;
  max_images_in_context: number;
  max_tokens: number;
  temperature: number;
  top_p: number;
  min_p: number;
  top_k: number;
  repetition_penalty: number;
  presence_penalty: number;
  frequency_penalty: number;
  seed: number;
  thought_syntax: string;
  vram_limit: number;
}

const DEFAULT_CONFIG: Config = {
  api_url: "http://localhost:8000/v1/chat/completions",
  model_name: "Qwen-VL",
  processing_mode: "Native Video (vLLM)",
  sampling_mode: "fps",
  interval: 2.0,
  target_fps: 1.0,
  max_frames_limit: 0,
  resolution_mode: "User Defined",
  image_width: 640,
  image_height: 480,
  system_prompt: "",
  interaction_mode: "Free-form",
  active_character: "",
  custom_mode: false,
  inject_thinking_tags: false,
  max_images_in_context: 3,
  max_tokens: 40960,
  temperature: 1.0,
  top_p: 0.95,
  min_p: 0.0,
  top_k: 20,
  repetition_penalty: 1.0,
  presence_penalty: 1.5,
  frequency_penalty: 0.0,
  seed: -1,
  thought_syntax: "<think>{content}</think>",
  vram_limit: 164000
};

interface BatchLogEntry {
  type: string;
  file?: string;
  index?: number;
  total?: number;
  completed?: number;
  skipped?: number;
  errors?: number;
  error?: string;
  caption_preview?: string;
  output?: string;
  existing?: number;
}

const THOUGHT_SYNTAXES = [
  { value: "<think>{content}</think>",       label: "Qwen3 / DeepSeek" },
  { value: "<|think|>{content}<|/think|>",   label: "Gemma 4" },
  { value: "<thinking>{content}</thinking>", label: "Claude-style" },
  { value: "",                               label: "None" },
  { value: "__custom__",                     label: "Custom..." },
];
const PRESET_VALUES = THOUGHT_SYNTAXES.filter(s => s.value !== "__custom__").map(s => s.value);

function App() {
  // State
  const [config, setConfig] = useState<Config>(DEFAULT_CONFIG);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [media, setMedia] = useState<UploadedMedia | null>(null);
  const [showPreview, setShowPreview] = useState(true);
  const [lastSentMedia, setLastSentMedia] = useState<string | null>(null);
  const [tokenEstimate, setTokenEstimate] = useState<TokenEstimate | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [maxModelLen, setMaxModelLen] = useState<number | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'offline'>('offline');

  // Tab state - left pane functions (chat is always visible on right)
  const [activeTab, setActiveTab] = useState<'captions' | 'batch' | 'prompts' | 'batch-review' | 'ui-prompts'>('batch');
  const [enablePaneContext, setEnablePaneContext] = useState(false);
  const [enableThinking, setEnableThinking] = useState(false);
  const [enableTools, setEnableTools] = useState(false);
  const [showThinking, setShowThinking] = useState(false);
  const [enableTTS, setEnableTTS] = useState(false);
  const [enableObservationPass, setEnableObservationPass] = useState(false);
  const [showObservation, setShowObservation] = useState(true);
  const [ttsLoadingIdx, setTtsLoadingIdx] = useState<number | null>(null);
  const [ttsPlayingIdx, setTtsPlayingIdx] = useState<number | null>(null);
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const ttsAbortRef = useRef<AbortController | null>(null);
  const [showCaptionPreview, setShowCaptionPreview] = useState(true);
  const [chatLogs, setChatLogs] = useState<{ filename: string; timestamp: string; title: string; message_count: number }[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // Prompt Manager state
  const [promptProfiles, setPromptProfiles] = useState<{ filename: string; name: string; is_default: boolean; description: string }[]>([]);
  const [activeProfile, setActiveProfile] = useState('prompts.json');
  const [promptData, setPromptData] = useState<any>(null);
  const [promptEditSection, setPromptEditSection] = useState<string>('chat_assistant');
  const [promptEditKey, setPromptEditKey] = useState<string>('system_prompt'); { }
  const [promptEditValue, setPromptEditValue] = useState('');
  const [promptSaveStatus, setPromptSaveStatus] = useState('');
  const [newProfileName, setNewProfileName] = useState('');

  // UI Agent prompts (config/modes.yaml) state
  const [uiModes, setUiModes] = useState<any>(null);
  const [uiModeKey, setUiModeKey] = useState<'free_form' | 'analytical' | 'roleplay'>('free_form');
  const [uiCharKey, setUiCharKey] = useState<string>('');
  const [newCharName, setNewCharName] = useState('');
  const [uiSaveStatus, setUiSaveStatus] = useState('');

  // Screenshot capture state
  const [pendingScreenshot, setPendingScreenshot] = useState<string | null>(null);
  // --- Live screen sharing ---
  const [liveSharing, setLiveSharing] = useState(false);
  const liveStreamRef = useRef<MediaStream | null>(null);
  const liveSharingRef = useRef(false);        // loop guard the async loop reads
  const liveObservationRef = useRef('');       // latest obs for chat body, no stale closure
  const functionPaneRef = useRef<HTMLDivElement>(null);

  // Batch captioner state
  const [batchDir, setBatchDir] = useState('');
  const [batchInstruction, setBatchInstruction] = useState('');
  const [captionTarget, setCaptionTarget] = useState('general');
  const [captionTargets, setCaptionTargets] = useState<{ id: string; name: string; description: string; style: string; token_limit: number | null; media_types: string[] }[]>([]);
  const [chatForceFps, setChatForceFps] = useState(true);
  const [batchForceFps, setBatchForceFps] = useState(true);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchLog, setBatchLog] = useState<BatchLogEntry[]>([]);
  const [batchProgress, setBatchProgress] = useState({ completed: 0, total: 0, skipped: 0, errors: 0 });
  const batchLogEndRef = useRef<HTMLDivElement>(null);

  // Caption reviewer state
  const [captionDir, setCaptionDir] = useState('');
  const [captionSubdirA, setCaptionSubdirA] = useState('pass1');
  const [captionSubdirB, setCaptionSubdirB] = useState('pass2');
  const [captionPairs, setCaptionPairs] = useState<DualCaptionPair[]>([]);
  const [captionIndex, setCaptionIndex] = useState(0);
  const [captionEditA, setCaptionEditA] = useState('');
  const [captionEditB, setCaptionEditB] = useState('');
  const [captionSaveStatus, setCaptionSaveStatus] = useState('');
  const [captionLoading, setCaptionLoading] = useState(false);
  const [rerunInstruction, setRerunInstruction] = useState('');
  const [rerunning, setRerunning] = useState(false);
  const [repassDir, setRepassDir] = useState('');
  const [repassInstruction, setRepassInstruction] = useState('');
  const [repassRunning, setRepassRunning] = useState(false);
  const [repassLog, setRepassLog] = useState<BatchLogEntry[]>([]);
  const [repassProgress, setRepassProgress] = useState({ completed: 0, total: 0, skipped: 0, errors: 0 });
  const [repassSkipMissing, setRepassSkipMissing] = useState(true);

  // Batch review state
  const [batchReviewDir, setBatchReviewDir] = useState('');
  const [batchReviewSubdirs, setBatchReviewSubdirs] = useState<BatchSubdir[]>([]);
  const [batchReviewLoading, setBatchReviewLoading] = useState(false);
  const [batchReviewStatus, setBatchReviewStatus] = useState('');
  const [batchReviewVersions, setBatchReviewVersions] = useState<Record<string, number>>({});
  const [batchReviewThumbSize, setBatchReviewThumbSize] = useState(180);

  // Caption reviewer image version (for cache-busting after rotate)
  const [captionImageVersion, setCaptionImageVersion] = useState(0);

  // Panel collapse states
  const [panels, setPanels] = useState({
    media: true,
    ffmpeg: false,
    sampling: true,
    inference: false,
    connection: false,
    mode: true
  });

  // Resizable sidebar
  const [sidebarWidth, setSidebarWidth] = useState(380);
  const [isResizing, setIsResizing] = useState(false);
  const [functionPaneWidth, setFunctionPaneWidth] = useState(500);
  const [isPaneResizing, setIsPaneResizing] = useState(false);
  const [functionPaneCollapsed, setFunctionPaneCollapsed] = useState(false);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // --- Live proactive loop coordination ---
  // messagesRef mirrors `messages` so the async loop reads fresh state without a
  // stale closure (same trick as the live* refs). genBusyRef is a shared lock so
  // a chat generation and an autonomous live-turn never run at once. userTurnSeqRef
  // bumps on every send so a late live-turn can tell a user took over and bow out.
  const messagesRef = useRef<Message[]>(messages);
  const genBusyRef = useRef(false);
  const liveTurnAbortRef = useRef<AbortController | null>(null);
  const userTurnSeqRef = useRef(0);
  const liveCooldownUntilRef = useRef(0);        // epoch ms; loop won't interject before this
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  // ----- TTS helpers -----
  const stopTTS = useCallback(() => {
    if (ttsAudioRef.current) {
      ttsAudioRef.current.pause();
      try { URL.revokeObjectURL(ttsAudioRef.current.src); } catch { /* ignore */ }
      ttsAudioRef.current = null;
    }
    if (ttsAbortRef.current) {
      ttsAbortRef.current.abort();
      ttsAbortRef.current = null;
    }
    setTtsLoadingIdx(null);
    setTtsPlayingIdx(null);
  }, []);

  const speakText = useCallback(async (text: string, msgIdx: number) => {
    stopTTS();
    if (!text || !text.trim()) return;
    setTtsLoadingIdx(msgIdx);
    const ctrl = new AbortController();
    ttsAbortRef.current = ctrl;
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, response_format: 'wav' }),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        const err = await res.text().catch(() => 'TTS error');
        console.warn('TTS request failed:', res.status, err);
        setTtsLoadingIdx(null);
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      ttsAudioRef.current = audio;
      audio.onended = () => {
        try { URL.revokeObjectURL(url); } catch { /* ignore */ }
        if (ttsAudioRef.current === audio) ttsAudioRef.current = null;
        setTtsPlayingIdx(p => p === msgIdx ? null : p);
      };
      audio.onerror = () => {
        try { URL.revokeObjectURL(url); } catch { /* ignore */ }
        if (ttsAudioRef.current === audio) ttsAudioRef.current = null;
        setTtsPlayingIdx(null);
      };
      setTtsLoadingIdx(null);
      setTtsPlayingIdx(msgIdx);
      await audio.play();
    } catch (e) {
      if (!(e instanceof Error && e.name === 'AbortError')) {
        console.warn('TTS error:', e);
      }
      setTtsLoadingIdx(null);
      setTtsPlayingIdx(null);
    }
  }, [stopTTS]);

  // Sidebar resize handlers
  const handleMouseDown = useCallback(() => {
    setIsResizing(true);
  }, []);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing) return;
    const newWidth = Math.max(280, Math.min(600, e.clientX));
    setSidebarWidth(newWidth);
  }, [isResizing]);

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);

  // Attach resize listeners
  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing, handleMouseMove, handleMouseUp]);

  // Function pane / chat pane resize handlers
  const handlePaneMouseDown = useCallback(() => {
    setIsPaneResizing(true);
  }, []);

  const handlePaneMouseMove = useCallback((e: MouseEvent) => {
    if (!isPaneResizing) return;
    const newWidth = Math.max(200, Math.min(900, e.clientX - sidebarWidth - 6));
    setFunctionPaneWidth(newWidth);
  }, [isPaneResizing, sidebarWidth]);

  const handlePaneMouseUp = useCallback(() => {
    setIsPaneResizing(false);
  }, []);

  useEffect(() => {
    if (isPaneResizing) {
      document.addEventListener('mousemove', handlePaneMouseMove);
      document.addEventListener('mouseup', handlePaneMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }
    return () => {
      document.removeEventListener('mousemove', handlePaneMouseMove);
      document.removeEventListener('mouseup', handlePaneMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isPaneResizing, handlePaneMouseMove, handlePaneMouseUp]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load config on mount, then probe models with the loaded URL
  useEffect(() => {
    fetchConfig().then(loaded => refreshModels(loaded.api_url));
    fetch('/api/caption-targets').then(r => r.json()).then(d => setCaptionTargets(d.targets || [])).catch(() => { });
    loadPromptProfiles();
    loadPromptProfile('prompts.json');
    loadUiModes();
  }, []);

  // Update token estimate when media or settings change
  useEffect(() => {
    if (media?.id) {
      updateTokenEstimate();
    }
  }, [media, config.target_fps, config.image_width, config.image_height, config.resolution_mode, maxModelLen]);

  const fetchConfig = async (): Promise<Config> => {
    try {
      const configRes = await fetch('/api/config');
      let merged = DEFAULT_CONFIG;
      if (configRes.ok) {
        const data = await configRes.json();
        merged = { ...DEFAULT_CONFIG, ...data };
      }
      // System prompt is sourced from prompt config (Prompts tab), not settings
      setConfig(merged);
      return merged;
    } catch {
      console.log('Using default config');
    }
    return DEFAULT_CONFIG;
  };

  const saveConfig = async () => {
    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config })
      });
    } catch (e) {
      console.error('Failed to save config:', e);
    }
  };

  const refreshModels = async (apiUrl?: string) => {
    const url = apiUrl || config.api_url;
    try {
      const res = await fetch(`/api/models?api_url=${encodeURIComponent(url)}`);
      const data = await res.json();
      if (data.status === 'connected') {
        setModels(data.models);
        setMaxModelLen(data.max_model_len);
        setConnectionStatus('connected');
        if (data.models.length > 0) {
          setConfig(c => {
            if (!data.models.includes(c.model_name)) {
              return { ...c, model_name: data.models[0] };
            }
            return c;
          });
        }
      } else {
        setConnectionStatus('offline');
      }
    } catch {
      setConnectionStatus('offline');
    }
  };

  const updateTokenEstimate = async () => {
    if (!media?.id) return;

    try {
      const params = new URLSearchParams({
        media_id: media.id,
        target_fps: config.target_fps.toString(),
        image_width: config.image_width.toString(),
        image_height: config.image_height.toString(),
        resolution_mode: config.resolution_mode,
        context_limit: config.vram_limit.toString(),
        ...(maxModelLen && { max_model_len: maxModelLen.toString() })
      });

      const res = await fetch(`/api/token-estimate?${params}`);
      const data = await res.json();
      setTokenEstimate(data);
    } catch {
      setTokenEstimate(null);
    }
  };

  const handleFileUpload = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();

        // Get thumbnail
        const thumbRes = await fetch(`/api/media/${data.id}/thumbnail`);
        const thumbData = await thumbRes.json();

        setMedia({
          ...data,
          thumbnail: thumbData.thumbnail
        });
      }
    } catch (e) {
      console.error('Upload failed:', e);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  }, []);

  const streamResponse = async (apiMessages: { role: string; content: string }[], includeMedia: boolean) => {
    abortControllerRef.current = new AbortController();
    genBusyRef.current = true;          // hold the shared lock: the live loop yields while we generate
    setIsLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: apiMessages,
          media_path: media?.path,
          include_media: includeMedia,
          max_tokens: config.max_tokens,
          temperature: config.temperature,
          top_p: config.top_p,
          min_p: config.min_p,
          top_k: config.top_k,
          repetition_penalty: config.repetition_penalty,
          presence_penalty: config.presence_penalty,
          frequency_penalty: config.frequency_penalty,
          seed: config.seed,
          enable_thinking: enableThinking,
          enable_observation_pass: enableObservationPass,
          interaction_mode: config.interaction_mode,
          active_character: config.active_character,
          // Agent prompt comes solely from modes.yaml (Roleplay: the character dict),
          // never from backend/config.json. Sending '' lets the backend fall through
          // to the active mode's text_prompt instead of a stale config.json system_prompt.
          system_prompt: '',
          inject_thinking: config.inject_thinking_tags,
          custom_mode: config.custom_mode,
          thought_syntax: config.thought_syntax,
          processing_mode: config.processing_mode,
          sampling_mode: config.sampling_mode,
          interval: config.interval,
          target_fps: config.target_fps,
          max_frames_limit: config.max_frames_limit,
          resolution_mode: config.resolution_mode,
          image_width: config.image_width,
          image_height: config.image_height,
          video_fps: config.target_fps,
          force_fps: chatForceFps,
          pane_context: getPaneContext() || undefined,
          live_observation: liveSharing ? (liveObservationRef.current || undefined) : undefined,
          tools_enabled: enableTools,
          save_thinking: true
        }),
        signal: abortControllerRef.current.signal
      });

      if (!res.ok) throw new Error('Chat request failed');

      const reader = res.body?.getReader();
      if (!reader) throw new Error('No reader available');

      let assistantContent = '';
      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') break;

            try {
              const parsed = JSON.parse(data);
              if (parsed.observation) {
                setMessages(prev => {
                  const newMsgs = [...prev];
                  const last = newMsgs[newMsgs.length - 1];
                  if (last && last.role === 'assistant') {
                    newMsgs[newMsgs.length - 1] = { ...last, observation: parsed.observation };
                  }
                  return newMsgs;
                });
              } else if (parsed.observation_error) {
                console.warn('Observation pass failed:', parsed.observation_error);
              } else if (parsed.tool_call) {
                const tc = parsed.tool_call;
                const argsStr = JSON.stringify(tc.arguments, null, 2);
                assistantContent += `\n\n**🔧 Tool Call:** \`${tc.name}\`\n\`\`\`json\n${argsStr}\n\`\`\`\n`;
                setMessages(prev => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1] = { ...newMsgs[newMsgs.length - 1], role: 'assistant', content: assistantContent };
                  return newMsgs;
                });
              } else if (parsed.tool_result) {
                const tr = parsed.tool_result;
                const resStr = tr.result.error
                  ? `❌ ${tr.result.error}`
                  : tr.name === 'read_file'
                    ? `✅ Read ${tr.result.size} bytes from \`${tr.result.path}\``
                    : tr.name === 'write_file'
                      ? `✅ Wrote ${tr.result.size} bytes to \`${tr.result.path}\``
                      : tr.name === 'view_media'
                      ? `✅ Viewed ${tr.result.media_type} \`${tr.result.path}\``
                    : tr.name === 'list_directory'
                        ? `✅ Found ${tr.result.count} entries in \`${tr.result.path}\``
                        : tr.name === 'web_search'
                          ? `✅ ${tr.result.results?.length || 0} results for "${tr.result.query}"${(tr.result.results || []).slice(0, 5).map((r: any) => `\n  - [${r.title}](${r.url})`).join('')}`
                          : tr.name === 'fetch_url'
                            ? `✅ Fetched ${tr.result.length} chars from \`${tr.result.url}\`${tr.result.truncated ? ' (truncated)' : ''}`
                            : `✅ ${JSON.stringify(tr.result)}`;
                assistantContent += `**Result:** ${resStr}\n\n`;
                setMessages(prev => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1] = { ...newMsgs[newMsgs.length - 1], role: 'assistant', content: assistantContent };
                  return newMsgs;
                });
              } else if (parsed.content) {
                assistantContent += parsed.content;
                setMessages(prev => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1] = { ...newMsgs[newMsgs.length - 1], role: 'assistant', content: assistantContent };
                  return newMsgs;
                });
              }
              if (parsed.error) throw new Error(parsed.error);
            } catch (parseErr) {
              if (parseErr instanceof Error && !(parseErr instanceof SyntaxError)) throw parseErr;
            }
          }
        }
      }
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') {
        // cancelled
      } else {
        const errorContent = `**Error:** ${e instanceof Error ? e.message : 'Unknown error'}`;
        setMessages(prev => {
          const newMsgs = [...prev];
          const lastMsg = newMsgs[newMsgs.length - 1];
          if (lastMsg && lastMsg.role === 'assistant' && !lastMsg.content) {
            newMsgs[newMsgs.length - 1] = { role: 'assistant', content: errorContent };
          } else {
            newMsgs.push({ role: 'assistant', content: errorContent });
          }
          return newMsgs;
        });
      }
    } finally {
      setIsLoading(false);
      genBusyRef.current = false;        // release the shared lock; live loop may resume
      abortControllerRef.current = null;
      setMessages(prev => {
        autoSaveChat(prev);
        // Auto-TTS: speak the just-completed assistant message if enabled
        if (enableTTS && prev.length > 0) {
          const last = prev[prev.length - 1];
          if (last.role === 'assistant' && last.content && !last.content.startsWith('**Error:')) {
            speakText(last.content, prev.length - 1);
          }
        }
        return prev;
      });
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    // User input always wins: invalidate any in-flight autonomous live-turn (it
    // self-discards on a changed sequence) and abort its request immediately.
    userTurnSeqRef.current++;
    liveTurnAbortRef.current?.abort();

    const userMessage = input.trim();
    const includeMedia = !!(media && media.id !== lastSentMedia);
    const hasScreenshot = !!pendingScreenshot;

    setMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      hasMedia: includeMedia,
      mediaId: includeMedia && media ? media.id : undefined,
      mediaType: includeMedia && media ? media.info?.media_type : undefined,
      screenshot: pendingScreenshot || undefined,
    }]);
    setInput('');

    if (includeMedia && media) setLastSentMedia(media.id);

    // Build API messages - for messages with screenshots, use multimodal content format
    const apiMessages: { role: string; content: any }[] = messages.map(m => {
      if (m.screenshot) {
        return {
          role: m.role,
          content: [
            { type: 'image_url', image_url: { url: m.screenshot } },
            { type: 'text', text: m.content },
          ],
        };
      }
      return { role: m.role, content: m.content };
    });

    if (hasScreenshot) {
      apiMessages.push({
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: pendingScreenshot } },
          { type: 'text', text: userMessage },
        ],
      });
    } else {
      apiMessages.push({ role: 'user', content: userMessage });
    }

    setPendingScreenshot(null);
    await streamResponse(apiMessages, includeMedia);
  };

  const handleRetry = async () => {
    if (isLoading || messages.length < 2) return;

    let lastUserIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') { lastUserIdx = i; break; }
    }
    if (lastUserIdx < 0) return;

    const retained = messages.slice(0, lastUserIdx + 1);
    setMessages(retained);

    const apiMessages = retained.map(m => ({ role: m.role, content: m.content }));
    const includeMedia = !!retained[lastUserIdx].hasMedia;

    await streamResponse(apiMessages, includeMedia);
  };

  const autoSaveChat = async (msgs: Message[]) => {
    if (msgs.length < 2) return;
    const firstUserMsg = msgs.find(m => m.role === 'user');
    const title = firstUserMsg ? firstUserMsg.content.slice(0, 80) : 'chat';
    const sampling = {
      seed: config.seed,
      temperature: config.temperature,
      top_p: config.top_p,
      top_k: config.top_k,
      min_p: config.min_p,
      repetition_penalty: config.repetition_penalty,
      presence_penalty: config.presence_penalty,
      frequency_penalty: config.frequency_penalty,
      enable_thinking: enableThinking,
      inject_thinking_tags: config.inject_thinking_tags,
      max_tokens: config.max_tokens,
    };
    try {
      await fetch('/api/chat/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: msgs, title, sampling })
      });
    } catch { /* silent */ }
  };

  const exportChat = () => {
    if (messages.length === 0) return;
    const lines = messages.map(m => {
      let header = `## ${m.role.toUpperCase()}`;
      if (m.screenshot) header += ' [📸 Screenshot]';
      if (m.hasMedia) header += ' [📎 Media]';
      return `${header}\n\n${m.content}`;
    });
    const md = `# Vision Lab Chat Export\n_${new Date().toLocaleString()}_\n\n${lines.join('\n\n---\n\n')}\n`;
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat_${new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-')}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportChatAsJSON = () => {
    if (messages.length === 0) return;
    const data = {
      timestamp: new Date().toISOString(),
      model: config.model_name,
      thinking_mode: enableThinking,
      messages: messages.map(m => ({
        role: m.role,
        content: m.content,
        has_media: m.hasMedia || false,
        has_screenshot: !!m.screenshot,
        autonomous: m.autonomous || false,
      })),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `reasoning_corpus_${new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleStop = () => {
    abortControllerRef.current?.abort();
    liveTurnAbortRef.current?.abort();
    genBusyRef.current = false;        // free the lock so the live loop can resume
    setIsLoading(false);
  };

  const clearChat = () => {
    setMessages([]);
    setLastSentMedia(null);
  };

  // --- Chat history (reload previous conversations) ---
  const loadChatLogs = async () => {
    try {
      const res = await fetch('/api/chat/logs');
      const data = await res.json();
      setChatLogs(data.logs || []);
    } catch { /* ignore */ }
  };

  const loadChat = async (filename: string) => {
    try {
      const res = await fetch(`/api/chat/logs/${encodeURIComponent(filename)}`);
      if (!res.ok) return;
      const data = await res.json();
      const restored: Message[] = (data.messages || []).map((m: any) => ({
        role: m.role,
        content: m.content || '',
        hasMedia: m.hasMedia || false,
        mediaId: m.mediaId,
        mediaType: m.mediaType,
        screenshot: m.screenshot,
        observation: m.observation,
        mediaCaption: m.mediaCaption,
        autonomous: m.autonomous,
      }));
      setMessages(restored);
      setLastSentMedia(null);   // uploaded media can't be re-attached; screenshots/observations are restored
      setShowHistory(false);
    } catch { /* ignore */ }
  };

  const deleteChatLog = async (filename: string) => {
    try {
      await fetch(`/api/chat/logs/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      setChatLogs(prev => prev.filter(l => l.filename !== filename));
    } catch { /* ignore */ }
  };

  const toggleHistory = () => {
    const next = !showHistory;
    setShowHistory(next);
    if (next) loadChatLogs();
  };

  const clearMedia = () => {
    setMedia(null);
    setLastSentMedia(null);
    setTokenEstimate(null);
  };

  // --- Caption Reviewer Functions ---

  const loadCaptionDir = async (dir: string) => {
    if (!dir.trim()) return;
    setCaptionLoading(true);
    setCaptionSaveStatus('');
    try {
      const params = new URLSearchParams({
        directory: dir,
        subdir_a: captionSubdirA || 'pass1',
        subdir_b: captionSubdirB || '',
      });
      const res = await fetch(`/api/captions/scan-dual?${params}`);
      if (!res.ok) throw new Error('Directory not found');
      const data = await res.json();
      setCaptionPairs(data.pairs);
      setCaptionIndex(0);
      if (data.pairs.length > 0) {
        setCaptionEditA(data.pairs[0].caption_a);
        setCaptionEditB(data.pairs[0].caption_b);
      }
    } catch (e) {
      setCaptionPairs([]);
      setCaptionIndex(0);
      setCaptionEditA('');
      setCaptionEditB('');
    } finally {
      setCaptionLoading(false);
    }
  };

  const navigateCaption = (direction: number) => {
    if (captionPairs.length === 0) return;
    const newIndex = (captionIndex + direction + captionPairs.length) % captionPairs.length;
    setCaptionIndex(newIndex);
    setCaptionEditA(captionPairs[newIndex].caption_a);
    setCaptionEditB(captionPairs[newIndex].caption_b);
    setCaptionSaveStatus('');
  };

  const rerunCaption = async (slot: 'a' | 'b') => {
    if (captionPairs.length === 0 || rerunning) return;
    const pair = captionPairs[captionIndex];
    const existingCaption = slot === 'a' ? captionEditA : captionEditB;
    const setEdit = slot === 'a' ? setCaptionEditA : setCaptionEditB;
    setRerunning(true);
    setCaptionSaveStatus('');
    setEdit('');
    let accumulated = '';
    try {
      const res = await fetch('/api/captions/rerun', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: pair.image_path,
          existing_caption: existingCaption,
          extra_instruction: rerunInstruction,
        })
      });
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
          if (!payload) continue;
          try {
            const msg = JSON.parse(payload);
            if (msg.token) {
              accumulated += msg.token;
              setEdit(accumulated);
            } else if (msg.done) {
              setEdit(msg.caption);
              accumulated = msg.caption;
            } else if (msg.error) {
              setCaptionSaveStatus(`Error: ${msg.error}`);
            }
          } catch { /* ignore */ }
        }
      }
    } catch (e) {
      setCaptionSaveStatus('Re-caption failed');
    } finally {
      setRerunning(false);
    }
  };

  const saveCaptionSlot = async (slot: 'a' | 'b') => {
    if (captionPairs.length === 0) return;
    const pair = captionPairs[captionIndex];
    const captionPath = slot === 'a' ? pair.caption_path_a : pair.caption_path_b;
    const caption = slot === 'a' ? captionEditA : captionEditB;
    if (!captionPath) return;
    try {
      const res = await fetch('/api/captions/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ caption_path: captionPath, caption })
      });
      if (res.ok) {
        setCaptionSaveStatus(`${slot.toUpperCase()} saved ✓`);
        setCaptionPairs(prev => {
          const updated = [...prev];
          if (slot === 'a') {
            updated[captionIndex] = { ...updated[captionIndex], caption_a: caption.trim(), has_caption_a: true };
          } else {
            updated[captionIndex] = { ...updated[captionIndex], caption_b: caption.trim(), has_caption_b: true };
          }
          return updated;
        });
      }
    } catch {
      setCaptionSaveStatus('Save failed');
    }
  };

  const deleteCaption = async () => {
    if (captionPairs.length === 0) return;
    const pair = captionPairs[captionIndex];
    // Delete the image; use caption_path_a as placeholder (backend deletes by image_path primarily)
    try {
      const res = await fetch('/api/captions/delete', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ caption_path: pair.caption_path_a, image_path: pair.image_path })
      });
      if (res.ok) {
        setCaptionPairs(prev => {
          const updated = [...prev];
          updated.splice(captionIndex, 1);
          return updated;
        });
        setCaptionIndex(prev => Math.max(0, Math.min(prev, captionPairs.length - 2)));
        setCaptionEditA('');
        setCaptionEditB('');
        setCaptionSaveStatus('Deleted ✓');
      }
    } catch {
      setCaptionSaveStatus('Delete failed');
    }
  };

  // --- Batch Review Functions ---

  const loadBatchReview = async (dir: string) => {
    if (!dir.trim()) return;
    setBatchReviewLoading(true);
    setBatchReviewStatus('');
    setBatchReviewSubdirs([]);
    try {
      const res = await fetch(`/api/captions/batch-scan?directory=${encodeURIComponent(dir)}`);
      if (res.ok) {
        const data = await res.json();
        setBatchReviewSubdirs(data.subdirs);
        setBatchReviewStatus(`${data.total_images} images in ${data.total_subdirs} folder${data.total_subdirs !== 1 ? 's' : ''}`);
      } else {
        setBatchReviewStatus('Directory not found');
      }
    } catch {
      setBatchReviewStatus('Load failed');
    } finally {
      setBatchReviewLoading(false);
    }
  };

  const deleteBatchImage = async (subdirIdx: number, imgIdx: number) => {
    const subdir = batchReviewSubdirs[subdirIdx];
    const img = subdir.images[imgIdx];
    try {
      const res = await fetch('/api/captions/delete', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_path: img.image_path, caption_path: img.caption_path })
      });
      if (res.ok) {
        setBatchReviewSubdirs(prev => {
          const updated = [...prev];
          const updatedImages = [...updated[subdirIdx].images];
          updatedImages.splice(imgIdx, 1);
          if (updatedImages.length === 0) {
            updated.splice(subdirIdx, 1);
          } else {
            updated[subdirIdx] = { ...updated[subdirIdx], images: updatedImages, total: updatedImages.length };
          }
          return updated;
        });
      }
    } catch {
      // silent fail — image stays in grid
    }
  };

  const rotateBatchImage = async (subdirIdx: number, imgIdx: number, degrees: number) => {
    const img = batchReviewSubdirs[subdirIdx].images[imgIdx];
    try {
      const res = await fetch('/api/captions/rotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_path: img.image_path, degrees })
      });
      if (res.ok) {
        setBatchReviewVersions(prev => ({ ...prev, [img.image_path]: (prev[img.image_path] || 0) + 1 }));
      }
    } catch { /* silent */ }
  };

  const rotateCaptionImage = async (degrees: number) => {
    if (!currentPair) return;
    try {
      const res = await fetch('/api/captions/rotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_path: currentPair.image_path, degrees })
      });
      if (res.ok) setCaptionImageVersion(prev => prev + 1);
    } catch { /* silent */ }
  };

  // --- Batch Re-caption Functions ---

  const startRepass = async () => {
    if (!repassDir.trim() || repassRunning) return;
    setRepassRunning(true);
    setRepassLog([]);
    setRepassProgress({ completed: 0, total: 0, skipped: 0, errors: 0 });
    try {
      const res = await fetch('/api/batch/recaption', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          directory: repassDir,
          extra_instruction: repassInstruction,
          skip_missing: repassSkipMissing,
          max_tokens: config.max_tokens,
          temperature: config.temperature,
          top_p: config.top_p,
          top_k: config.top_k,
          presence_penalty: config.presence_penalty,
          enable_thinking: enableThinking,
          strip_thinking: true,
        }),
      });
      const reader = res.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          try {
            const entry: BatchLogEntry = JSON.parse(line.slice(6));
            setRepassLog(prev => [...prev, entry]);
            if (entry.type === 'start') {
              setRepassProgress(p => ({ ...p, total: entry.total || 0 }));
            } else if (entry.type === 'done') {
              setRepassProgress(p => ({ ...p, completed: entry.completed || p.completed + 1 }));
            } else if (entry.type === 'skip') {
              setRepassProgress(p => ({ ...p, skipped: p.skipped + 1 }));
            } else if (entry.type === 'error') {
              setRepassProgress(p => ({ ...p, errors: p.errors + 1 }));
            } else if (entry.type === 'complete' || entry.type === 'stopped') {
              setRepassRunning(false);
            }
          } catch { /* skip parse errors */ }
        }
      }
    } catch (e) {
      setRepassLog(prev => [...prev, { type: 'error', error: String(e) }]);
    } finally {
      setRepassRunning(false);
    }
  };

  const stopRepass = async () => {
    try { await fetch('/api/batch/recaption/stop', { method: 'POST' }); } catch { /* ignore */ }
  };

  // --- Batch Captioner Functions ---

  const startBatch = async () => {
    if (!batchDir.trim() || batchRunning) return;
    setBatchRunning(true);
    setBatchLog([]);
    setBatchProgress({ completed: 0, total: 0, skipped: 0, errors: 0 });

    try {
      const res = await fetch('/api/batch/caption', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          directory: batchDir,
          instruction: batchInstruction,
          caption_target: captionTarget,
          max_tokens: config.max_tokens,
          temperature: config.temperature,
          top_p: config.top_p,
          top_k: config.top_k,
          presence_penalty: config.presence_penalty,
          enable_thinking: enableThinking,
          video_fps: config.target_fps,
          force_fps: batchForceFps,
          strip_thinking: true,
          skip_existing: true,
        }),
      });

      const reader = res.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          try {
            const entry: BatchLogEntry = JSON.parse(line.slice(6));
            setBatchLog(prev => [...prev, entry]);
            if (entry.type === 'start') {
              setBatchProgress(p => ({ ...p, total: entry.total || 0 }));
            } else if (entry.type === 'done') {
              setBatchProgress(p => ({ ...p, completed: (entry.completed || p.completed + 1) }));
            } else if (entry.type === 'skip') {
              setBatchProgress(p => ({ ...p, skipped: p.skipped + 1 }));
            } else if (entry.type === 'error') {
              setBatchProgress(p => ({ ...p, errors: p.errors + 1 }));
            } else if (entry.type === 'complete' || entry.type === 'stopped') {
              setBatchRunning(false);
            }
          } catch { /* skip parse errors */ }
        }
      }
    } catch (e) {
      setBatchLog(prev => [...prev, { type: 'error', error: String(e) }]);
    } finally {
      setBatchRunning(false);
    }
  };

  const stopBatch = async () => {
    try { await fetch('/api/batch/stop', { method: 'POST' }); } catch { /* ignore */ }
  };

  // --- Prompt Manager Functions ---

  const loadPromptProfiles = async () => {
    try {
      const res = await fetch('/api/prompt-profiles');
      const data = await res.json();
      setPromptProfiles(data.profiles || []);
    } catch { /* ignore */ }
  };

  const loadPromptProfile = async (filename: string) => {
    try {
      const res = await fetch(`/api/prompt-profiles/${encodeURIComponent(filename)}`);
      if (!res.ok) return;
      const data = await res.json();
      setPromptData(data);
      setActiveProfile(filename);
      setPromptSaveStatus('');
      selectPromptField(data, promptEditSection, promptEditKey);
    } catch { /* ignore */ }
  };

  const selectPromptField = (data: any, section: string, key: string) => {
    if (!data) return;
    const sectionData = data[section];
    if (!sectionData) { setPromptEditValue(''); return; }
    const val = sectionData[key];
    setPromptEditValue(typeof val === 'string' ? val : JSON.stringify(val, null, 2));
    setPromptEditSection(section);
    setPromptEditKey(key);
  };

  const updatePromptField = () => {
    if (!promptData) return;
    const updated = { ...promptData };
    if (!updated[promptEditSection]) updated[promptEditSection] = {};
    try {
      updated[promptEditSection][promptEditKey] = JSON.parse(promptEditValue);
    } catch {
      updated[promptEditSection][promptEditKey] = promptEditValue;
    }
    setPromptData(updated);
    setPromptSaveStatus('Modified (unsaved)');
  };

  const saveCurrentProfile = async () => {
    if (!promptData) return;
    try {
      const res = await fetch('/api/prompt-profiles/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: activeProfile, data: promptData }),
      });
      if (res.ok) {
        setPromptSaveStatus('Saved');
        loadPromptProfiles();
      }
    } catch { setPromptSaveStatus('Save failed'); }
  };

  const saveAsNewProfile = async () => {
    if (!promptData || !newProfileName.trim()) return;
    const filename = `prompts_${newProfileName.trim().replace(/\s+/g, '_')}`;
    try {
      const res = await fetch('/api/prompt-profiles/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, data: promptData }),
      });
      if (res.ok) {
        const result = await res.json();
        setActiveProfile(result.filename);
        setNewProfileName('');
        setPromptSaveStatus('Saved as new profile');
        loadPromptProfiles();
      }
    } catch { setPromptSaveStatus('Save failed'); }
  };

  const activateProfile = async (filename: string) => {
    try {
      await fetch(`/api/prompt-profiles/activate/${encodeURIComponent(filename)}`, { method: 'POST' });
      setPromptSaveStatus(`Activated: ${filename}`);
      fetch('/api/caption-targets').then(r => r.json()).then(d => setCaptionTargets(d.targets || [])).catch(() => { });
    } catch { setPromptSaveStatus('Activation failed'); }
  };

  const deleteProfile = async (filename: string) => {
    try {
      await fetch(`/api/prompt-profiles/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      loadPromptProfiles();
      if (activeProfile === filename) loadPromptProfile('prompts.json');
    } catch { /* ignore */ }
  };

  const getEditableSections = (): { section: string; keys: string[] }[] => {
    if (!promptData) return [];
    const sections: { section: string; keys: string[] }[] = [];
    for (const [section, val] of Object.entries(promptData)) {
      if (section.startsWith('_')) continue;
      if (typeof val !== 'object' || val === null) continue;
      const keys = Object.keys(val as object).filter(k => !k.startsWith('_'));
      if (keys.length > 0) sections.push({ section, keys });
    }
    return sections;
  };

  // --- UI Agent prompts (config/modes.yaml) ---
  const loadUiModes = async () => {
    try {
      const res = await fetch('/api/ui-modes');
      const data = await res.json();
      const m = data.ui_modes || {};
      setUiModes(m);
      const rpActive = m.roleplay?.active_character || '';
      if (rpActive) setUiCharKey(rpActive);
    } catch { /* ignore */ }
  };

  const getUiField = (field: string): string => {
    const mode = uiModes?.[uiModeKey] || {};
    if (field === 'media_image') return mode.media_prompt?.image ?? '';
    if (field === 'media_video') return mode.media_prompt?.video ?? '';
    if (field === 'text_prompt' && uiModeKey === 'roleplay') {
      return mode.characters?.[uiCharKey]?.text_prompt ?? '';
    }
    return mode[field] ?? '';
  };

  const updateUiField = (field: string, value: string) => {
    setUiModes((prev: any) => {
      const next = JSON.parse(JSON.stringify(prev || {}));
      if (!next[uiModeKey]) next[uiModeKey] = {};
      if (field === 'media_image' || field === 'media_video') {
        if (!next[uiModeKey].media_prompt) next[uiModeKey].media_prompt = {};
        next[uiModeKey].media_prompt[field === 'media_image' ? 'image' : 'video'] = value;
      } else if (field === 'text_prompt' && uiModeKey === 'roleplay') {
        if (!uiCharKey) return prev;
        if (!next.roleplay.characters) next.roleplay.characters = {};
        if (!next.roleplay.characters[uiCharKey]) next.roleplay.characters[uiCharKey] = {};
        next.roleplay.characters[uiCharKey].text_prompt = value;
      } else {
        next[uiModeKey][field] = value;
      }
      return next;
    });
    setUiSaveStatus('Modified (unsaved)');
  };

  const addCharacter = () => {
    const name = newCharName.trim();
    if (!name) return;
    setUiModes((prev: any) => {
      const next = JSON.parse(JSON.stringify(prev || {}));
      if (!next.roleplay) next.roleplay = { interaction_mode: 'Roleplay', observation_prompt: '', media_prompt: { image: '', video: '' } };
      if (!next.roleplay.characters) next.roleplay.characters = {};
      if (!next.roleplay.characters[name]) next.roleplay.characters[name] = { text_prompt: '' };
      return next;
    });
    setUiCharKey(name);
    setNewCharName('');
    setUiSaveStatus('Modified (unsaved)');
  };

  const deleteCharacter = () => {
    if (!uiCharKey) return;
    setUiModes((prev: any) => {
      const next = JSON.parse(JSON.stringify(prev || {}));
      if (next.roleplay?.characters) delete next.roleplay.characters[uiCharKey];
      return next;
    });
    setUiCharKey('');
    setUiSaveStatus('Modified (unsaved)');
  };

  const saveUiModes = async () => {
    if (!uiModes) return;
    try {
      const payload = JSON.parse(JSON.stringify(uiModes));
      if (payload.roleplay) {
        payload.roleplay.active_character = config.active_character || payload.roleplay.active_character || '';
      }
      const res = await fetch('/api/ui-modes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ui_modes: payload }),
      });
      if (res.ok) { setUiModes(payload); setUiSaveStatus('Saved ✓'); }
      else setUiSaveStatus('Save failed');
    } catch { setUiSaveStatus('Save failed'); }
  };

  const getPaneContext = (): string => {
    if (!enablePaneContext) return '';
    if (activeTab === 'batch') {
      const { completed, total, skipped, errors } = batchProgress;
      const lastDone = [...batchLog].reverse().find(e => e.type === 'done');
      const targetName = captionTargets.find(t => t.id === captionTarget)?.name || captionTarget;
      return `[Batch Captioner] Target: ${targetName}, Directory: ${batchDir || 'not set'}, Progress: ${completed}/${total} (${skipped} skipped, ${errors} errors), Status: ${batchRunning ? 'running' : 'idle'}${lastDone ? `, Last caption: "${lastDone.caption_preview?.slice(0, 100)}..."` : ''}`;
    }
    if (activeTab === 'captions' && currentPair) {
      return `[Caption Reviewer] File: ${currentPair.filename} (${captionIndex + 1}/${captionPairs.length}), A: ${currentPair.has_caption_a ? 'captioned' : 'missing'}, B: ${currentPair.has_caption_b ? 'captioned' : 'missing'}`;
    }
    return '';
  };

  const captureScreenshot = async () => {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
      const track = stream.getVideoTracks()[0];
      const imageCapture = new ImageCapture(track);
      const bitmap = await (imageCapture as any).grabFrame();
      track.stop();

      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext('2d')!.drawImage(bitmap, 0, 0);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      setPendingScreenshot(dataUrl);
    } catch (e) {
      console.error('Screenshot capture failed:', e);
    }
  };

  const clearScreenshot = () => setPendingScreenshot(null);

  // --- Live screen sharing ---
  // Hold a getDisplayMedia stream open and run the proactive loop: each cycle
  // captures a short frame-burst and sends it (with a bounded slice of the
  // conversation) to /api/live-turn, which returns OBS (rolling silent context,
  // rides along with chat as `live_observation`) + SAY (an optional unprompted
  // interjection). User turns always win via the shared genBusyRef lock.
  const captureFrameFromStream = async (stream: MediaStream): Promise<string | null> => {
    const track = stream.getVideoTracks()[0];
    if (!track || track.readyState !== 'live') return null;
    try {
      const imageCapture = new (window as any).ImageCapture(track);
      const bitmap = await imageCapture.grabFrame();
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext('2d')!.drawImage(bitmap, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.7);
    } catch {
      return null;
    }
  };

  const stopLiveShare = () => {
    liveSharingRef.current = false;
    setLiveSharing(false);
    liveStreamRef.current?.getTracks().forEach(t => t.stop());
    liveStreamRef.current = null;
  };

  const delay = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

  // Grab a short chronological burst of frames — a "clip" the model reads as
  // motion (important for watching video). ~n frames spread across spanMs.
  const captureBurst = async (stream: MediaStream, n: number, spanMs: number): Promise<string[]> => {
    const frames: string[] = [];
    const gap = n > 1 ? Math.floor(spanMs / (n - 1)) : 0;
    for (let i = 0; i < n; i++) {
      const f = await captureFrameFromStream(stream);
      if (f) frames.push(f);
      if (i < n - 1 && liveSharingRef.current) await delay(gap);
    }
    return frames;
  };

  // Push an UNPROMPTED assistant turn (the model "spoke" on its own) into chat.
  const appendAutonomousTurn = (text: string) => {
    const idx = messagesRef.current.length;          // index the new turn will land at
    setMessages(prev => {
      const next: Message[] = [...prev, { role: 'assistant', content: text, autonomous: true }];
      autoSaveChat(next);
      return next;
    });
    if (enableTTS) speakText(text, idx);
  };

  // Tuning (mirrors backend DEFAULT_CONFIG live_* keys).
  const LIVE_CONTEXT_TURNS = 6;        // recent messages the loop is allowed to send
  const LIVE_CLIP_FRAMES = 8;          // frames per clip
  const LIVE_CLIP_SPAN_MS = 3500;      // clip duration window
  const LIVE_COOLDOWN_MS = 15000;      // min gap between unprompted interjections

  // The proactive live loop. Each cycle: yield if a generation is running, grab a
  // clip, then one merged /api/live-turn call returns OBS (silent context) + SAY
  // (interject or [SILENT]). User turns always win — see the seq/lock checks.
  const runLiveLoop = async () => {
    while (liveSharingRef.current && liveStreamRef.current) {
      // Yield to any in-flight generation (user chat, or a prior live-turn).
      if (genBusyRef.current) { await delay(400); continue; }

      // Capture first (cheap, no model call). A user may send during this.
      const frames = await captureBurst(liveStreamRef.current, LIVE_CLIP_FRAMES, LIVE_CLIP_SPAN_MS);
      if (!liveSharingRef.current) break;          // stopped mid-capture
      if (frames.length === 0) break;              // track ended/lost
      if (genBusyRef.current) continue;            // a user send slipped in during capture

      // Claim the lock and record which user-turn era we belong to.
      genBusyRef.current = true;
      const seq = userTurnSeqRef.current;
      liveTurnAbortRef.current = new AbortController();
      try {
        const recent = messagesRef.current.slice(-LIVE_CONTEXT_TURNS).map(m => ({
          role: m.role,
          content: m.content.replace(/<think>[\s\S]*?<\/think>/gi, '').trim().slice(0, 500),
        }));
        const res = await fetch('/api/live-turn', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            frames,
            recent_messages: recent,
            interaction_mode: config.interaction_mode,
            active_character: config.active_character,
          }),
          signal: liveTurnAbortRef.current.signal,
        });
        if (res.ok) {
          const data = await res.json();
          if (data.observation) liveObservationRef.current = data.observation;
          // Speak only if: the model chose to, no user turn started meanwhile,
          // and we're past the cooldown.
          const fresh = userTurnSeqRef.current === seq;
          const cooled = Date.now() >= liveCooldownUntilRef.current;
          if (data.interjection && fresh && cooled) {
            appendAutonomousTurn(data.interjection);
            liveCooldownUntilRef.current = Date.now() + LIVE_COOLDOWN_MS;
          }
        }
      } catch {
        // aborted (user took over) or transient — just continue
      } finally {
        // Release ONLY if a user turn didn't seize the lock during our request;
        // otherwise streamResponse now owns it and will release it itself.
        if (userTurnSeqRef.current === seq) genBusyRef.current = false;
        liveTurnAbortRef.current = null;
      }
    }
  };

  const startLiveShare = async () => {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
      liveStreamRef.current = stream;
      // Stop cleanly if the user ends sharing from the browser's own control bar
      stream.getVideoTracks()[0]?.addEventListener('ended', stopLiveShare);
      liveSharingRef.current = true;
      setLiveSharing(true);
      runLiveLoop();
    } catch (e) {
      console.error('Live screen share failed to start:', e);
    }
  };

  const currentPair = captionPairs[captionIndex] || null;
  const captionStats = {
    total: captionPairs.length,
    withCaptionsA: captionPairs.filter(p => p.has_caption_a).length,
    withCaptionsB: captionPairs.filter(p => p.has_caption_b).length,
    slotA: {
      charCount: captionEditA.length,
      wordCount: captionEditA.trim() ? captionEditA.trim().split(/\s+/).length : 0,
      tokenCount: Math.ceil(captionEditA.length / 4),
    },
    slotB: {
      charCount: captionEditB.length,
      wordCount: captionEditB.trim() ? captionEditB.trim().split(/\s+/).length : 0,
      tokenCount: Math.ceil(captionEditB.length / 4),
    },
  };

  const togglePanel = (panel: keyof typeof panels) => {
    setPanels(p => ({ ...p, [panel]: !p[panel] }));
  };

  // Render helpers
  const renderPanel = (
    id: keyof typeof panels,
    icon: React.ReactNode,
    title: string,
    children: React.ReactNode
  ) => (
    <div className="panel">
      <div className="panel-header" onClick={() => togglePanel(id)}>
        <span className="panel-title">
          {icon}
          {title}
        </span>
        {panels[id] ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
      </div>
      <div className={`panel-content ${panels[id] ? '' : 'collapsed'}`}>
        {children}
      </div>
    </div>
  );

  const renderTokenStatus = () => {
    if (!tokenEstimate || tokenEstimate.status === 'no_media') {
      return (
        <div className="token-status">
          <div className="token-status-header">Token Status</div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            {tokenEstimate?.message || 'Upload media to estimate...'}
          </p>
        </div>
      );
    }

    const { visual_tokens = 0, remaining = 0, context_limit = 0, status } = tokenEstimate;
    const usedPercent = context_limit > 0 ? ((context_limit - remaining) / context_limit) * 100 : 0;

    return (
      <div className="token-status">
        <div className="token-status-header">
          Token Status ({tokenEstimate.media_type})
        </div>
        <div className="token-bar">
          <div
            className={`token-bar-fill ${status}`}
            style={{ width: `${Math.min(usedPercent, 100)}%` }}
          />
        </div>
        <div className="token-stats">
          <span className="token-value">
            Visual: {visual_tokens.toLocaleString()}
          </span>
          <span className={`token-value ${status === 'danger' ? 'danger' : ''}`}>
            Remaining: {remaining.toLocaleString()}
          </span>
          <span className="token-limit">
            / {context_limit.toLocaleString()}
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className={`app-container ${isResizing || isPaneResizing ? 'resizing' : ''}`}>
      {/* Sidebar */}
      <aside className="sidebar" style={{ width: sidebarWidth, minWidth: sidebarWidth }}>
        <div className="sidebar-header">
          <h1><Eye size={24} /> Vision Lab</h1>
          <p>Multi-Modal VLM Interface v3.2</p>
        </div>

        <div className="sidebar-content">
          {/* Media Upload */}
          {renderPanel('media', <Upload size={16} />, 'Media Upload', (
            <>
              {!media ? (
                <div
                  className="file-upload"
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="file-upload-icon" />
                  <p className="file-upload-text">
                    Drop files here or <span>browse</span>
                  </p>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8 }}>
                    Images: JPG, PNG, GIF, WebP • Videos: MP4, AVI, MOV, MKV
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*,video/*"
                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                  />
                </div>
              ) : (

                <div className="media-preview">
                  {media.thumbnail && (
                    <div className="media-preview-container" style={{ width: '100%' }}>
                      {/* 1. Toggleable Image */}
                      {showPreview && (
                        <img
                          src={media.thumbnail}
                          alt="Preview"
                          style={{ marginBottom: 8, borderRadius: '4px', width: '100%' }}
                        />
                      )}

                      {/* 2. The Hide/Show Button */}
                      <button
                        className="btn btn-secondary btn-icon"
                        onClick={() => setShowPreview(!showPreview)}
                        style={{
                          width: '100%',
                          marginBottom: 12,
                          fontSize: '0.75rem',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '8px'
                        }}
                      >
                        {showPreview ? <EyeOff size={14} /> : <Eye size={14} />}
                        {showPreview ? 'Hide Preview' : 'Show Preview'}
                      </button>
                    </div>
                  )}

                  {/* 3. Media Stats (Always visible unless media is cleared) */}
                  <div className="media-info">
                    <div className="media-info-grid">
                      <div className="media-info-item">
                        <span className="media-info-label">Type</span>
                        <span>{media.info.media_type === 'video' ? '🎬 Video' : '🖼️ Image'}</span>
                      </div>
                      <div className="media-info-item">
                        <span className="media-info-label">Size</span>
                        <span>{media.info.res}</span>
                      </div>
                      {media.info.media_type === 'video' && (
                        <>
                          <div className="media-info-item">
                            <span className="media-info-label">Duration</span>
                            <span>{media.info.dur?.toFixed(1)}s</span>
                          </div>
                          <div className="media-info-item">
                            <span className="media-info-label">FPS</span>
                            <span>{media.info.fps?.toFixed(1)}</span>
                          </div>
                        </>
                      )}
                    </div>

                    <button
                      className="btn btn-secondary btn-icon"
                      onClick={clearMedia}
                      style={{ marginTop: 8, width: '100%' }}
                    >
                      <X size={16} /> Remove
                    </button>
                  </div>
                </div>
              )}
              
              <div style={{ marginTop: 12 }}>
                {renderTokenStatus()}
              </div>
            </>
          ))}

          {/* FFmpeg Processing (Video only) */}
              {media?.info.media_type === 'video' && renderPanel('ffmpeg', <Scissors size={16} />, 'FFmpeg Pre-Process', (
                <>
                  <div className="row">
                    <div className="form-group">
                      <label className="form-label">Start (s)</label>
                      <input type="number" className="form-input" defaultValue={0} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">End (s)</label>
                      <input
                        type="number"
                        className="form-input"
                        defaultValue={media.info.dur || 60}
                      />
                    </div>
                  </div>
                  <div className="row">
                    <div className="form-group">
                      <label className="form-label">Width</label>
                      <input type="number" className="form-input" defaultValue={640} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Height</label>
                      <input type="number" className="form-input" defaultValue={480} />
                    </div>
                  </div>
                  <button className="btn btn-secondary" style={{ width: '100%' }}>
                    <Scissors size={16} /> Generate Clip
                  </button>
                </>
              ))}

              {/* Frame Sampling */}
              {renderPanel('sampling', <Film size={16} />, 'Frame Sampling', (
                <>
                  <div className="form-group">
                    <label className="form-label">Processing Mode</label>
                    <div className="form-radio-group">
                      {['Native Video (vLLM)', 'Extraction'].map(mode => (
                        <label
                          key={mode}
                          className={`form-radio ${config.processing_mode === mode ? 'active' : ''}`}
                        >
                          <input
                            type="radio"
                            checked={config.processing_mode === mode}
                            onChange={() => setConfig(c => ({ ...c, processing_mode: mode }))}
                          />
                          {mode}
                        </label>
                      ))}
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Sampling Method</label>
                    <div className="form-radio-group">
                      {['fps', 'interval'].map(mode => (
                        <label
                          key={mode}
                          className={`form-radio ${config.sampling_mode === mode ? 'active' : ''}`}
                        >
                          <input
                            type="radio"
                            checked={config.sampling_mode === mode}
                            onChange={() => setConfig(c => ({ ...c, sampling_mode: mode }))}
                          />
                          {mode.toUpperCase()}
                        </label>
                      ))}
                    </div>
                  </div>

                  {config.sampling_mode === 'fps' ? (
                    <div className="form-group">
                      <label className="form-label">Target FPS</label>
                      <div className="form-slider-container">
                        <input
                          type="range"
                          className="form-slider"
                          min="0.1"
                          max="30"
                          step="0.1"
                          value={config.target_fps}
                          onChange={(e) => setConfig(c => ({ ...c, target_fps: parseFloat(e.target.value) }))}
                        />
                        <span className="form-slider-value">{config.target_fps.toFixed(1)}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="form-group">
                      <label className="form-label">Interval (s)</label>
                      <input
                        type="number"
                        className="form-input"
                        value={config.interval}
                        onChange={(e) => setConfig(c => ({ ...c, interval: parseFloat(e.target.value) }))}
                      />
                    </div>
                  )}

                  <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <input
                      type="checkbox"
                      id="chat-force-fps"
                      checked={chatForceFps}
                      onChange={(e) => setChatForceFps(e.target.checked)}
                    />
                    <label htmlFor="chat-force-fps" className="form-label" style={{ margin: 0 }}>
                      Force FPS ({config.target_fps.toFixed(1)})
                    </label>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Resolution Mode</label>
                    <div className="form-radio-group">
                      {['Native Resolution', 'User Defined'].map(mode => (
                        <label
                          key={mode}
                          className={`form-radio ${config.resolution_mode === mode ? 'active' : ''}`}
                        >
                          <input
                            type="radio"
                            checked={config.resolution_mode === mode}
                            onChange={() => setConfig(c => ({ ...c, resolution_mode: mode }))}
                          />
                          {mode}
                        </label>
                      ))}
                    </div>
                  </div>

                  {config.resolution_mode === 'User Defined' && (
                    <div className="row">
                      <div className="form-group">
                        <label className="form-label">Width</label>
                        <input
                          type="number"
                          className="form-input"
                          value={config.image_width}
                          onChange={(e) => setConfig(c => ({ ...c, image_width: parseInt(e.target.value) }))}
                        />
                      </div>
                      <div className="form-group">
                        <label className="form-label">Height</label>
                        <input
                          type="number"
                          className="form-input"
                          value={config.image_height}
                          onChange={(e) => setConfig(c => ({ ...c, image_height: parseInt(e.target.value) }))}
                        />
                      </div>
                    </div>
                  )}
                </>
              ))}

              {/* Interaction Mode */}
              {renderPanel('mode', <Brain size={16} />, 'Interaction Mode', (
                <>
                  <div className="form-group">
                    <label className="form-label">Mode</label>
                    <div className="form-radio-group">
                      {['Free-form', 'Analytical', 'Roleplay'].map(mode => (
                        <label
                          key={mode}
                          className={`form-radio ${config.interaction_mode === mode ? 'active' : ''}`}
                        >
                          <input
                            type="radio"
                            checked={config.interaction_mode === mode}
                            onChange={() => setConfig(c => ({ ...c, interaction_mode: mode }))}
                          />
                          {mode}
                        </label>
                      ))}
                    </div>
                  </div>

                  {config.interaction_mode === 'Roleplay' && (
                    <div className="form-group">
                      <label className="form-label">Character</label>
                      <select
                        className="form-select"
                        value={config.active_character}
                        onChange={(e) => setConfig(c => ({ ...c, active_character: e.target.value }))}
                      >
                        <option value="">(mode default)</option>
                        {Object.keys(uiModes?.roleplay?.characters || {}).map(name => (
                          <option key={name} value={name}>{name}</option>
                        ))}
                      </select>
                    </div>
                  )}

                  <div className="row">
                    <label className={`form-checkbox ${config.inject_thinking_tags ? 'active' : ''}`}>
                      <input
                        type="checkbox"
                        checked={config.inject_thinking_tags}
                        onChange={(e) => setConfig(c => ({ ...c, inject_thinking_tags: e.target.checked }))}
                      />
                      <span className="checkbox-indicator"><Check size={12} /></span>
                      Thinking Tags
                    </label>
                    <label className={`form-checkbox ${config.custom_mode ? 'active' : ''}`}>
                      <input
                        type="checkbox"
                        checked={config.custom_mode}
                        onChange={(e) => setConfig(c => ({ ...c, custom_mode: e.target.checked }))}
                      />
                      <span className="checkbox-indicator"><Check size={12} /></span>
                      Custom Mode
                    </label>
                  </div>
                </>
              ))}

              {/* Inference Settings */}
              {renderPanel('inference', <Zap size={16} />, 'Inference Tuning', (
                <>
                  <div className="form-group">
                    <label className="form-label">Max Tokens</label>
                    <div className="form-slider-container">
                      <input
                        type="range"
                        className="form-slider"
                        min="512"
                        max="131072"
                        step="1024"
                        value={config.max_tokens}
                        onChange={(e) => setConfig(c => ({ ...c, max_tokens: parseInt(e.target.value) }))}
                      />
                      <span className="form-slider-value">{config.max_tokens.toLocaleString()}</span>
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Temperature</label>
                    <div className="form-slider-container">
                      <input
                        type="range"
                        className="form-slider"
                        min="0"
                        max="2"
                        step="0.05"
                        value={config.temperature}
                        onChange={(e) => setConfig(c => ({ ...c, temperature: parseFloat(e.target.value) }))}
                      />
                      <span className="form-slider-value">{config.temperature.toFixed(2)}</span>
                    </div>
                  </div>

                  <div className="row">
                    <div className="form-group">
                      <label className="form-label">Top P</label>
                      <div className="form-slider-container">
                        <input
                          type="range"
                          className="form-slider"
                          min="0"
                          max="1"
                          step="0.05"
                          value={config.top_p}
                          onChange={(e) => setConfig(c => ({ ...c, top_p: parseFloat(e.target.value) }))}
                        />
                        <span className="form-slider-value">{config.top_p.toFixed(2)}</span>
                      </div>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Min P</label>
                      <div className="form-slider-container">
                        <input
                          type="range"
                          className="form-slider"
                          min="0"
                          max="1"
                          step="0.01"
                          value={config.min_p}
                          onChange={(e) => setConfig(c => ({ ...c, min_p: parseFloat(e.target.value) }))}
                        />
                        <span className="form-slider-value">{config.min_p.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Top K</label>
                    <div className="form-slider-container">
                      <input
                        type="range"
                        className="form-slider"
                        min="1"
                        max="200"
                        step="1"
                        value={config.top_k}
                        onChange={(e) => setConfig(c => ({ ...c, top_k: parseInt(e.target.value) }))}
                      />
                      <span className="form-slider-value">{config.top_k}</span>
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Repetition Penalty</label>
                    <div className="form-slider-container">
                      <input
                        type="range"
                        className="form-slider"
                        min="1"
                        max="2"
                        step="0.05"
                        value={config.repetition_penalty}
                        onChange={(e) => setConfig(c => ({ ...c, repetition_penalty: parseFloat(e.target.value) }))}
                      />
                      <span className="form-slider-value">{config.repetition_penalty.toFixed(2)}</span>
                    </div>
                  </div>

                  <div className="row">
                    <div className="form-group">
                      <label className="form-label">Presence</label>
                      <div className="form-slider-container">
                        <input
                          type="range"
                          className="form-slider"
                          min="0"
                          max="2"
                          step="0.1"
                          value={config.presence_penalty}
                          onChange={(e) => setConfig(c => ({ ...c, presence_penalty: parseFloat(e.target.value) }))}
                        />
                        <span className="form-slider-value">{config.presence_penalty.toFixed(1)}</span>
                      </div>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Frequency</label>
                      <div className="form-slider-container">
                        <input
                          type="range"
                          className="form-slider"
                          min="0"
                          max="2"
                          step="0.1"
                          value={config.frequency_penalty}
                          onChange={(e) => setConfig(c => ({ ...c, frequency_penalty: parseFloat(e.target.value) }))}
                        />
                        <span className="form-slider-value">{config.frequency_penalty.toFixed(1)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Seed (-1 = random)</label>
                    <input
                      type="number"
                      className="form-input"
                      value={config.seed}
                      onChange={(e) => setConfig(c => ({ ...c, seed: parseInt(e.target.value) }))}
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Thought Syntax</label>
                    <select
                      className="form-select"
                      value={PRESET_VALUES.includes(config.thought_syntax) ? config.thought_syntax : "__custom__"}
                      onChange={(e) => {
                        if (e.target.value !== "__custom__")
                          setConfig(c => ({ ...c, thought_syntax: e.target.value }))
                      }}
                    >
                      {THOUGHT_SYNTAXES.map(s => (
                        <option key={s.value} value={s.value}>{s.label}</option>
                      ))}
                    </select>
                    {!PRESET_VALUES.includes(config.thought_syntax) && (
                      <input
                        type="text"
                        className="form-input"
                        style={{ marginTop: '6px' }}
                        placeholder="e.g. <|think|>{content}<|/think|>"
                        value={config.thought_syntax}
                        onChange={(e) => setConfig(c => ({ ...c, thought_syntax: e.target.value }))}
                      />
                    )}
                  </div>

                  <div className="form-group">
                    <label className="form-label">Max Images in Context</label>
                    <div className="form-slider-container">
                      <input
                        type="range"
                        className="form-slider"
                        min="1"
                        max="10"
                        step="1"
                        value={config.max_images_in_context}
                        onChange={(e) => setConfig(c => ({ ...c, max_images_in_context: parseInt(e.target.value) }))}
                      />
                      <span className="form-slider-value">{config.max_images_in_context}</span>
                    </div>
                  </div>
                </>
              ))}

              {/* Connection */}
              {renderPanel('connection', <Cpu size={16} />, 'Connection', (
                <>
                  <div className="form-group">
                    <label className="form-label">API URL</label>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <input
                        type="text"
                        className="form-input"
                        value={config.api_url}
                        onChange={(e) => setConfig(c => ({ ...c, api_url: e.target.value }))}
                      />
                      <button
                        className="btn btn-secondary btn-icon"
                        onClick={() => refreshModels()}
                        title="Refresh models"
                      >
                        <RefreshCw size={16} />
                      </button>
                    </div>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Model</label>
                    <select
                      className="form-select"
                      value={config.model_name}
                      onChange={(e) => setConfig(c => ({ ...c, model_name: e.target.value }))}
                    >
                      {models.length > 0 ? (
                        models.map(m => <option key={m} value={m}>{m}</option>)
                      ) : (
                        <option value={config.model_name}>{config.model_name}</option>
                      )}
                    </select>
                  </div>

                  <button
                    className="btn btn-primary"
                    style={{ width: '100%' }}
                    onClick={saveConfig}
                  >
                    <Settings size={16} /> Save Settings
                  </button>
                </>
              ))}
            </div >
      </aside>

      {/* Resizable Divider */}
      <div
        className={`resize-handle ${isResizing ? 'active' : ''}`}
        onMouseDown={handleMouseDown}
      />

      {/* Main Content Area - Dual Pane */}
      <main className="main-content">
        {/* LEFT PANE: Function Tabs */}
        {!functionPaneCollapsed && (
        <>
        <div className="function-pane" ref={functionPaneRef} style={{ width: functionPaneWidth, flex: 'none' }}>
          <div className="tab-bar">
            <button
              className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`}
              onClick={() => setActiveTab('batch')}
            >
              <Film size={16} /> Batch Caption
            </button>
            <button
              className={`tab-btn ${activeTab === 'captions' ? 'active' : ''}`}
              onClick={() => setActiveTab('captions')}
            >
              <FileText size={16} /> Caption Review
            </button>
            <button
              className={`tab-btn ${activeTab === 'batch-review' ? 'active' : ''}`}
              onClick={() => setActiveTab('batch-review')}
            >
              <ImageIcon size={16} /> Batch Review
            </button>
            <button
              className={`tab-btn ${activeTab === 'prompts' ? 'active' : ''}`}
              onClick={() => setActiveTab('prompts')}
            >
              <Settings size={16} /> Prompts
            </button>
            <button
              className={`tab-btn ${activeTab === 'ui-prompts' ? 'active' : ''}`}
              onClick={() => setActiveTab('ui-prompts')}
            >
              <Brain size={16} /> UI Prompts
            </button>
            <button
              className="tab-collapse-btn"
              onClick={() => setFunctionPaneCollapsed(true)}
              title="Hide tabs panel"
            >
              <ChevronLeft size={16} />
            </button>
          </div>

          {activeTab === 'batch' ? (
            /* Batch Captioner */
            <div className="batch-panel">
              <div className="batch-config">
                <div className="form-group">
                  <label className="form-label">Media Directory</label>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <input
                      type="text"
                      className="form-input"
                      placeholder="/path/to/media/"
                      value={batchDir}
                      onChange={(e) => setBatchDir(e.target.value)}
                      style={{ flex: 1 }}
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label className="form-label">Caption Target</label>
                  <select
                    className="form-select"
                    value={captionTarget}
                    onChange={(e) => setCaptionTarget(e.target.value)}
                  >
                    {captionTargets.map(t => (
                      <option key={t.id} value={t.id}>{t.name}</option>
                    ))}
                  </select>
                  {captionTargets.find(t => t.id === captionTarget) && (
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
                      {captionTargets.find(t => t.id === captionTarget)?.description}
                      {captionTargets.find(t => t.id === captionTarget)?.token_limit && (
                        <> — <span style={{ color: 'var(--accent-primary)' }}>
                          {captionTargets.find(t => t.id === captionTarget)?.token_limit} token limit
                        </span></>
                      )}
                    </p>
                  )}
                </div>
                <div className="form-group">
                  <label className="form-label">Additional Instruction (optional)</label>
                  <textarea
                    className="form-input"
                    rows={2}
                    placeholder="Appended to target instruction. Leave empty to use target defaults."
                    value={batchInstruction}
                    onChange={(e) => setBatchInstruction(e.target.value)}
                  />
                </div>
                <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    type="checkbox"
                    id="batch-thinking"
                    checked={enableThinking}
                    onChange={(e) => setEnableThinking(e.target.checked)}
                  />
                  <label htmlFor="batch-thinking" className="form-label" style={{ margin: 0 }}>Qwen3.5 Thinking Mode</label>
                </div>
                <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    type="checkbox"
                    id="batch-force-fps"
                    checked={batchForceFps}
                    onChange={(e) => setBatchForceFps(e.target.checked)}
                  />
                  <label htmlFor="batch-force-fps" className="form-label" style={{ margin: 0 }}>
                    Force FPS ({config.target_fps.toFixed(1)})
                  </label>
                </div>
                <div className="row">
                  {!batchRunning ? (
                    <button className="btn btn-primary" style={{ flex: 1 }} onClick={startBatch} disabled={!batchDir.trim()}>
                      <Zap size={16} /> Start Batch
                    </button>
                  ) : (
                    <button className="btn btn-danger" style={{ flex: 1 }} onClick={stopBatch}>
                      <X size={16} /> Stop
                    </button>
                  )}
                </div>
                {batchProgress.total > 0 && (
                  <>
                    <div className="batch-progress-bar">
                      <div
                        className="batch-progress-bar-fill"
                        style={{ width: `${((batchProgress.completed + batchProgress.skipped) / batchProgress.total) * 100}%` }}
                      />
                    </div>
                  </>
                )}
              </div>

              <div className="batch-progress">
                {batchLog.map((entry, i) => (
                  <div key={i} className={`batch-progress-item ${entry.type}`}>
                    {entry.type === 'processing' && <><RefreshCw size={12} className="spinner" /> Captioning {entry.file}</>}
                    {entry.type === 'done' && <><Check size={12} /> {entry.file}</>}
                    {entry.type === 'skip' && <><ChevronRight size={12} /> {entry.file} (skipped)</>}
                    {entry.type === 'error' && <><X size={12} /> {entry.file}: {entry.error}</>}
                    {entry.type === 'start' && <><Zap size={12} /> Starting: {entry.total} files ({entry.existing} existing)</>}
                    {entry.type === 'complete' && <><Check size={12} /> Complete: {entry.completed} captioned, {entry.skipped} skipped, {entry.errors} errors</>}
                    {entry.type === 'stopped' && <><X size={12} /> Stopped at {entry.completed}/{entry.total}</>}
                  </div>
                ))}
                <div ref={batchLogEndRef} />
              </div>

              {batchProgress.total > 0 && (
                <div className="batch-summary">
                  <span className="batch-summary-stat"><Check size={14} /> {batchProgress.completed}</span>
                  <span className="batch-summary-stat"><ChevronRight size={14} /> {batchProgress.skipped} skipped</span>
                  <span className="batch-summary-stat">{batchProgress.errors > 0 && <><X size={14} /> {batchProgress.errors} errors</>}</span>
                  <span style={{ marginLeft: 'auto', color: 'var(--text-muted)' }}>/ {batchProgress.total} total</span>
                </div>
              )}

              {/* Re-caption Pass */}
              <div className="batch-repass-section">
                <div className="batch-repass-header">
                  <RefreshCw size={14} />
                  <span>Re-caption Pass</span>
                  <span className="batch-repass-hint">Runs each existing caption + image back through the model</span>
                </div>
                <div className="batch-config">
                  <div className="form-group">
                    <label className="form-label">Directory</label>
                    <input
                      type="text"
                      className="form-input"
                      placeholder="/path/to/dataset/"
                      value={repassDir}
                      onChange={(e) => setRepassDir(e.target.value)}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Extra Instructions (optional)</label>
                    <textarea
                      className="form-input"
                      rows={2}
                      placeholder="e.g. 'Expand detail on lighting and composition. Keep under 120 words.'"
                      value={repassInstruction}
                      onChange={(e) => setRepassInstruction(e.target.value)}
                    />
                  </div>
                  <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                    <input
                      type="checkbox"
                      id="repass-skip-missing"
                      checked={repassSkipMissing}
                      onChange={(e) => setRepassSkipMissing(e.target.checked)}
                    />
                    <label htmlFor="repass-skip-missing" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                      Skip images without existing captions
                    </label>
                  </div>
                  <div className="row">
                    {!repassRunning ? (
                      <button className="btn btn-secondary" style={{ flex: 1 }} onClick={startRepass} disabled={!repassDir.trim()}>
                        <RefreshCw size={16} /> Start Re-caption Pass
                      </button>
                    ) : (
                      <button className="btn btn-danger" style={{ flex: 1 }} onClick={stopRepass}>
                        <X size={16} /> Stop
                      </button>
                    )}
                  </div>
                  {repassProgress.total > 0 && (
                    <div className="batch-progress-bar">
                      <div
                        className="batch-progress-bar-fill"
                        style={{ width: `${((repassProgress.completed + repassProgress.skipped) / repassProgress.total) * 100}%` }}
                      />
                    </div>
                  )}
                </div>
                {repassLog.length > 0 && (
                  <div className="batch-progress">
                    {repassLog.map((entry, i) => (
                      <div key={i} className={`batch-progress-item ${entry.type}`}>
                        {entry.type === 'processing' && <><RefreshCw size={12} className="spinner" /> Re-captioning {entry.file}</>}
                        {entry.type === 'done' && <><Check size={12} /> {entry.file}</>}
                        {entry.type === 'skip' && <><ChevronRight size={12} /> {entry.file} (no caption, skipped)</>}
                        {entry.type === 'error' && <><X size={12} /> {entry.file}: {entry.error}</>}
                        {entry.type === 'start' && <><Zap size={12} /> Starting re-caption: {entry.total} images</>}
                        {entry.type === 'complete' && <><Check size={12} /> Complete: {entry.completed} re-captioned, {entry.skipped} skipped, {entry.errors} errors</>}
                        {entry.type === 'stopped' && <><X size={12} /> Stopped at {entry.completed}/{entry.total}</>}
                      </div>
                    ))}
                  </div>
                )}
                {repassProgress.total > 0 && (
                  <div className="batch-summary">
                    <span className="batch-summary-stat"><Check size={14} /> {repassProgress.completed}</span>
                    <span className="batch-summary-stat"><ChevronRight size={14} /> {repassProgress.skipped} skipped</span>
                    <span className="batch-summary-stat">{repassProgress.errors > 0 && <><X size={14} /> {repassProgress.errors} errors</>}</span>
                    <span style={{ marginLeft: 'auto', color: 'var(--text-muted)' }}>/ {repassProgress.total} total</span>
                  </div>
                )}
              </div>
            </div>
          ) : activeTab === 'captions' ? (
            /* Caption Review Tab */
            <div className="caption-review">
              <div className="caption-review-toolbar">
                <div style={{ display: 'flex', gap: 8, flex: 1 }}>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Image base directory (e.g., /path/to/dataset/batch1)"
                    value={captionDir}
                    onChange={(e) => setCaptionDir(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') loadCaptionDir(captionDir); }}
                    style={{ flex: 1 }}
                  />
                  <button
                    className="btn btn-primary"
                    onClick={() => loadCaptionDir(captionDir)}
                    disabled={captionLoading}
                  >
                    <FolderOpen size={16} /> Load
                  </button>
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Pass A subdir"
                    value={captionSubdirA}
                    onChange={(e) => setCaptionSubdirA(e.target.value)}
                    style={{ width: 120, fontSize: '0.8rem' }}
                    title="Subdirectory name for caption pass A (e.g. pass1)"
                  />
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Pass B subdir"
                    value={captionSubdirB}
                    onChange={(e) => setCaptionSubdirB(e.target.value)}
                    style={{ width: 120, fontSize: '0.8rem' }}
                    title="Subdirectory name for caption pass B (e.g. pass2)"
                  />
                </div>
                {captionPairs.length > 0 && (
                  <div className="caption-review-stats">
                    <span>{captionStats.total} images</span>
                    <span>{captionSubdirA}: {captionStats.withCaptionsA} done</span>
                    {captionSubdirB && <span>{captionSubdirB}: {captionStats.withCaptionsB} done</span>}
                  </div>
                )}
              </div>

              {captionPairs.length === 0 ? (
                <div className="empty-state">
                  <ImageIcon className="empty-state-icon" />
                  <h3 className="empty-state-title">Caption Reviewer</h3>
                  <p className="empty-state-text">
                    Enter a base image directory and subdir names for two caption passes, then click Load.
                    Captions are stored as <code>base/pass1/stem.txt</code> and <code>base/pass2/stem.txt</code>.
                  </p>
                </div>
              ) : (
                <div className="caption-review-content">
                  <div className="caption-review-image-panel">
                    {showCaptionPreview && (
                      <img
                        src={`/api/captions/image?path=${encodeURIComponent(currentPair?.image_path || '')}&v=${captionImageVersion}`}
                        alt={currentPair?.filename || ''}
                        className="caption-review-image"
                      />
                    )}
                    <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
                      <button
                        className="btn btn-secondary"
                        onClick={() => rotateCaptionImage(90)}
                        title="Rotate 90° counter-clockwise"
                        style={{ flex: 1, fontSize: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}
                      >
                        <RotateCcw size={13} /> CCW
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={() => rotateCaptionImage(-90)}
                        title="Rotate 90° clockwise"
                        style={{ flex: 1, fontSize: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}
                      >
                        <RotateCw size={13} /> CW
                      </button>
                    </div>
                    <button
                      className="btn btn-secondary"
                      onClick={() => setShowCaptionPreview(p => !p)}
                      style={{ width: '100%', marginBottom: 8, fontSize: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
                    >
                      {showCaptionPreview ? <EyeOff size={14} /> : <Eye size={14} />}
                      {showCaptionPreview ? 'Hide Preview' : 'Show Preview'}
                    </button>
                    <div className="caption-review-filename">
                      {currentPair?.filename}
                    </div>
                    {captionSaveStatus && (
                      <span className="caption-save-status" style={{ marginTop: 4, display: 'block', textAlign: 'center' }}>{captionSaveStatus}</span>
                    )}
                    <button
                      className="btn btn-danger"
                      onClick={deleteCaption}
                      disabled={!currentPair}
                      title="Delete image file"
                      style={{ width: '100%', marginTop: 8, fontSize: '0.75rem' }}
                    >
                      <Trash2 size={14} /> Delete Image
                    </button>
                  </div>

                  {/* Pass A */}
                  <div className="caption-review-edit-panel">
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ background: 'var(--accent)', color: '#000', borderRadius: 3, padding: '1px 6px' }}>A</span>
                      {captionSubdirA || 'pass1'}
                      {currentPair && !currentPair.has_caption_a && <span className="caption-missing-badge">missing</span>}
                    </div>
                    <textarea
                      className="caption-review-textarea"
                      value={captionEditA}
                      onChange={(e) => { setCaptionEditA(e.target.value); setCaptionSaveStatus(''); }}
                      placeholder="Pass A caption..."
                    />
                    <div className="caption-rerun-row">
                      <input
                        type="text"
                        className="form-input"
                        placeholder="Extra re-caption instructions (optional)..."
                        value={rerunInstruction}
                        onChange={(e) => setRerunInstruction(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter') rerunCaption('a'); }}
                        disabled={rerunning}
                        style={{ flex: 1, fontSize: '0.8rem' }}
                      />
                      <button
                        className={`btn btn-secondary ${rerunning ? 'loading' : ''}`}
                        onClick={() => rerunCaption('a')}
                        disabled={rerunning}
                        title="Re-caption into Pass A"
                      >
                        <RefreshCw size={14} /> {rerunning ? 'Running…' : 'Re-caption'}
                      </button>
                    </div>
                    <div className="caption-review-edit-footer">
                      <div className="caption-review-counts">
                        <span>{captionStats.slotA.charCount} chars</span>
                        <span>~{captionStats.slotA.wordCount} words</span>
                        <span>~{captionStats.slotA.tokenCount} tok</span>
                      </div>
                      <button className="btn btn-primary" onClick={() => saveCaptionSlot('a')}>
                        <Save size={16} /> Save A
                      </button>
                    </div>
                  </div>

                  {/* Pass B */}
                  {captionSubdirB && (
                    <div className="caption-review-edit-panel">
                      <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ background: '#7c3aed', color: '#fff', borderRadius: 3, padding: '1px 6px' }}>B</span>
                        {captionSubdirB}
                        {currentPair && !currentPair.has_caption_b && <span className="caption-missing-badge">missing</span>}
                      </div>
                      <textarea
                        className="caption-review-textarea"
                        value={captionEditB}
                        onChange={(e) => { setCaptionEditB(e.target.value); setCaptionSaveStatus(''); }}
                        placeholder="Pass B caption..."
                      />
                      <div className="caption-rerun-row">
                        <input
                          type="text"
                          className="form-input"
                          placeholder="Extra re-caption instructions (optional)..."
                          value={rerunInstruction}
                          onChange={(e) => setRerunInstruction(e.target.value)}
                          onKeyDown={(e) => { if (e.key === 'Enter') rerunCaption('b'); }}
                          disabled={rerunning}
                          style={{ flex: 1, fontSize: '0.8rem' }}
                        />
                        <button
                          className={`btn btn-secondary ${rerunning ? 'loading' : ''}`}
                          onClick={() => rerunCaption('b')}
                          disabled={rerunning}
                          title="Re-caption into Pass B"
                        >
                          <RefreshCw size={14} /> {rerunning ? 'Running…' : 'Re-caption'}
                        </button>
                      </div>
                      <div className="caption-review-edit-footer">
                        <div className="caption-review-counts">
                          <span>{captionStats.slotB.charCount} chars</span>
                          <span>~{captionStats.slotB.wordCount} words</span>
                          <span>~{captionStats.slotB.tokenCount} tok</span>
                        </div>
                        <button className="btn btn-primary" onClick={() => saveCaptionSlot('b')}>
                          <Save size={16} /> Save B
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {captionPairs.length > 0 && (
                <div className="caption-review-nav">
                  <button className="btn btn-secondary" onClick={() => navigateCaption(-1)}>
                    <ChevronLeft size={16} /> Previous
                  </button>
                  <span className="caption-review-counter">
                    {captionIndex + 1} / {captionPairs.length}
                  </span>
                  <button className="btn btn-secondary" onClick={() => navigateCaption(1)}>
                    Next <ChevronRight size={16} />
                  </button>
                </div>
              )}
            </div>
          ) : activeTab === 'batch-review' ? (
            /* Batch Review Tab */
            <div className="batch-review-panel">
              <div className="batch-review-toolbar">
                <input
                  type="text"
                  className="form-input"
                  placeholder="Root directory (e.g., /path/to/dataset)"
                  value={batchReviewDir}
                  onChange={(e) => setBatchReviewDir(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') loadBatchReview(batchReviewDir); }}
                  style={{ flex: 1 }}
                />
                <button
                  className="btn btn-primary"
                  onClick={() => loadBatchReview(batchReviewDir)}
                  disabled={batchReviewLoading}
                >
                  <FolderOpen size={16} /> {batchReviewLoading ? 'Loading…' : 'Load'}
                </button>
                <div className="batch-review-size-control">
                  <ImageIcon size={12} />
                  <input
                    type="range"
                    min={100}
                    max={400}
                    step={20}
                    value={batchReviewThumbSize}
                    onChange={(e) => setBatchReviewThumbSize(Number(e.target.value))}
                    title={`Thumbnail size: ${batchReviewThumbSize}px`}
                  />
                  <ImageIcon size={18} />
                </div>
              </div>
              {batchReviewStatus && (
                <div className="batch-review-stats">{batchReviewStatus}</div>
              )}
              {batchReviewSubdirs.length === 0 && !batchReviewLoading ? (
                <div className="empty-state">
                  <ImageIcon className="empty-state-icon" />
                  <h3 className="empty-state-title">Batch Review</h3>
                  <p className="empty-state-text">
                    Enter a root directory to browse all images across subdirectories. Click delete to remove an image and its caption file.
                  </p>
                </div>
              ) : (
                <div className="batch-review-scroll">
                  {batchReviewSubdirs.map((subdir, si) => (
                    <div key={subdir.dir} className="batch-review-subdir">
                      <div className="batch-review-subdir-header">
                        <FolderOpen size={13} style={{ marginRight: 6, verticalAlign: 'middle' }} />
                        {subdir.rel_dir}
                        <span style={{ marginLeft: 8, opacity: 0.6 }}>({subdir.total})</span>
                      </div>
                      <div className="batch-review-grid" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${batchReviewThumbSize}px, 1fr))` }}>
                        {subdir.images.map((img, ii) => (
                          <div key={img.image_path} className="batch-review-card">
                            <img
                              src={`/api/captions/image?path=${encodeURIComponent(img.image_path)}&v=${batchReviewVersions[img.image_path] || 0}`}
                              alt={img.filename}
                              loading="lazy"
                            />
                            <div className="batch-review-card-footer">
                              <span className="batch-review-card-name" title={img.filename}>
                                {img.filename}
                              </span>
                              <button
                                className="batch-review-card-btn"
                                onClick={() => rotateBatchImage(si, ii, 90)}
                                title="Rotate CCW"
                              >
                                <RotateCcw size={14} />
                              </button>
                              <button
                                className="batch-review-card-btn"
                                onClick={() => rotateBatchImage(si, ii, -90)}
                                title="Rotate CW"
                              >
                                <RotateCw size={14} />
                              </button>
                              <button
                                className="batch-review-card-delete"
                                onClick={() => deleteBatchImage(si, ii)}
                                title="Delete image and caption"
                              >
                                <Trash2 size={14} />
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : activeTab === 'prompts' ? (
            /* Prompt Manager Tab */
            <div className="batch-panel">
              <div className="batch-config">
                <div className="form-group">
                  <label className="form-label">Profile</label>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <select
                      className="form-select"
                      value={activeProfile}
                      onChange={(e) => loadPromptProfile(e.target.value)}
                      style={{ flex: 1 }}
                    >
                      {promptProfiles.map(p => (
                        <option key={p.filename} value={p.filename}>
                          {p.name}{p.is_default ? ' (default)' : ''}
                        </option>
                      ))}
                    </select>
                    <button
                      className="btn btn-secondary btn-icon"
                      onClick={() => activateProfile(activeProfile)}
                      title="Set as active profile (copies to default)"
                      disabled={activeProfile === 'prompts.json'}
                    >
                      <Check size={16} />
                    </button>
                    <button
                      className="btn btn-secondary btn-icon"
                      onClick={() => deleteProfile(activeProfile)}
                      title="Delete this profile"
                      disabled={activeProfile === 'prompts.json'}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">Section / Field</label>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <select
                      className="form-select"
                      value={`${promptEditSection}::${promptEditKey}`}
                      onChange={(e) => {
                        const [s, k] = e.target.value.split('::');
                        setPromptEditSection(s);
                        setPromptEditKey(k);
                        selectPromptField(promptData, s, k);
                      }}
                      style={{ flex: 1 }}
                    >
                      {getEditableSections().map(({ section, keys }) => (
                        keys.map(key => (
                          <option key={`${section}::${key}`} value={`${section}::${key}`}>
                            {section} → {key}
                          </option>
                        ))
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <textarea
                className="caption-review-textarea"
                value={promptEditValue}
                onChange={(e) => { setPromptEditValue(e.target.value); setPromptSaveStatus(''); }}
                placeholder="Select a field to edit..."
                style={{ flex: 1 }}
              />

              <div className="batch-summary" style={{ flexDirection: 'column', gap: 8, alignItems: 'stretch' }}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button className="btn btn-secondary" onClick={updatePromptField} style={{ flex: 1 }}>
                    Apply Edit
                  </button>
                  <button className="btn btn-primary" onClick={saveCurrentProfile} style={{ flex: 1 }}>
                    <Save size={14} /> Save Profile
                  </button>
                  {promptSaveStatus && (
                    <span style={{ fontSize: '0.8rem', color: 'var(--success)' }}>{promptSaveStatus}</span>
                  )}
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="New profile name..."
                    value={newProfileName}
                    onChange={(e) => setNewProfileName(e.target.value)}
                    style={{ flex: 1 }}
                  />
                  <button
                    className="btn btn-secondary"
                    onClick={saveAsNewProfile}
                    disabled={!newProfileName.trim()}
                  >
                    Save As New
                  </button>
                </div>
              </div>
            </div>
          ) : activeTab === 'ui-prompts' ? (
            /* UI Agent Prompts (config/modes.yaml) */
            <div className="batch-panel">
              <div className="batch-config">
                <div className="form-group">
                  <label className="form-label">Mode</label>
                  <select
                    className="form-select"
                    value={uiModeKey}
                    onChange={(e) => setUiModeKey(e.target.value as 'free_form' | 'analytical' | 'roleplay')}
                  >
                    <option value="free_form">Free-form</option>
                    <option value="analytical">Analytical</option>
                    <option value="roleplay">Roleplay</option>
                  </select>
                </div>

                {uiModeKey === 'roleplay' && (
                  <div className="form-group">
                    <label className="form-label">Character (editing)</label>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <select
                        className="form-select"
                        style={{ flex: 1 }}
                        value={uiCharKey}
                        onChange={(e) => setUiCharKey(e.target.value)}
                      >
                        <option value="">(select a character)</option>
                        {Object.keys(uiModes?.roleplay?.characters || {}).map(name => (
                          <option key={name} value={name}>{name}</option>
                        ))}
                      </select>
                      <input
                        type="text"
                        className="form-input"
                        placeholder="New character..."
                        value={newCharName}
                        onChange={(e) => setNewCharName(e.target.value)}
                        style={{ flex: 1 }}
                      />
                      <button className="btn btn-secondary" onClick={addCharacter} disabled={!newCharName.trim()}>
                        Add
                      </button>
                      <button className="btn btn-secondary btn-icon" onClick={deleteCharacter} disabled={!uiCharKey} title="Delete character">
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>
                )}
              </div>

              <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12, padding: '0 4px' }}>
                <div className="form-group">
                  <label className="form-label">
                    Text Prompt{uiModeKey === 'roleplay' ? ` — ${uiCharKey || 'no character selected'}` : ''}
                  </label>
                  <textarea
                    className="caption-review-textarea"
                    style={{ minHeight: 120 }}
                    value={getUiField('text_prompt')}
                    disabled={uiModeKey === 'roleplay' && !uiCharKey}
                    placeholder={uiModeKey === 'roleplay' ? 'Select or add a character to edit its prompt...' : 'System / identity prompt for this mode...'}
                    onChange={(e) => updateUiField('text_prompt', e.target.value)}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Observation Prompt (Observe pass)</label>
                  <textarea
                    className="caption-review-textarea"
                    style={{ minHeight: 90 }}
                    value={getUiField('observation_prompt')}
                    placeholder="Pass A instruction when Observe is ON. Empty = fall back to config.json default."
                    onChange={(e) => updateUiField('observation_prompt', e.target.value)}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Media Prompt — Image (Observe OFF)</label>
                  <textarea
                    className="caption-review-textarea"
                    style={{ minHeight: 70 }}
                    value={getUiField('media_image')}
                    placeholder="Prepended to a normal turn when an image is attached."
                    onChange={(e) => updateUiField('media_image', e.target.value)}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Media Prompt — Video (Observe OFF)</label>
                  <textarea
                    className="caption-review-textarea"
                    style={{ minHeight: 70 }}
                    value={getUiField('media_video')}
                    placeholder="Prepended to a normal turn when a video is attached."
                    onChange={(e) => updateUiField('media_video', e.target.value)}
                  />
                </div>
              </div>

              <div className="batch-summary" style={{ alignItems: 'center', gap: 8 }}>
                <button className="btn btn-primary" onClick={saveUiModes} style={{ flex: 1 }}>
                  <Save size={14} /> Save UI Prompts
                </button>
                {uiSaveStatus && (
                  <span style={{ fontSize: '0.8rem', color: 'var(--success)' }}>{uiSaveStatus}</span>
                )}
              </div>
            </div>
          ) : null}
        </div>

        {/* Pane Resize Handle */}
        <div
          className={`resize-handle ${isPaneResizing ? 'active' : ''}`}
          onMouseDown={handlePaneMouseDown}
        />
        </>
        )}

        {functionPaneCollapsed && (
          <button
            className="function-pane-reopen"
            onClick={() => setFunctionPaneCollapsed(false)}
            title="Show tabs panel"
          >
            <ChevronRight size={16} />
          </button>
        )}

        {/* RIGHT PANE: Persistent Chat */}
        <div className="chat-pane">
          {/* Chat Header */}
          <div className="chat-header">
            <div className="chat-header-left">
              <MessageSquare size={20} />
              <span className="chat-model-badge">{config.model_name}</span>
              <span className="chat-model-badge" style={{
                background: config.interaction_mode === 'Free-form' ? 'var(--accent-glow)' :
                  config.interaction_mode === 'Analytical' ? 'rgba(90, 200, 250, 0.15)' :
                    'rgba(255, 159, 10, 0.15)',
                borderColor: config.interaction_mode === 'Free-form' ? 'var(--accent-primary)' :
                  config.interaction_mode === 'Analytical' ? 'var(--info)' :
                    'var(--warning)'
              }}>
                {config.interaction_mode}
              </span>
            </div>
            <div className="chat-status">
              <span className={`chat-status-dot ${connectionStatus}`} />
              {connectionStatus === 'connected' ? 'Connected' : 'Offline'}
            </div>
          </div>

          {/* Messages */}
          <div className="messages-area">
            {showHistory && (
              <div style={{
                border: '1px solid var(--border, #333)',
                borderRadius: 8,
                margin: '0 0 12px',
                background: 'var(--bg-secondary, #1a1a1a)',
                maxHeight: 260,
                overflowY: 'auto',
              }}>
                <div style={{
                  padding: '8px 12px', fontSize: '0.85rem', fontWeight: 600,
                  borderBottom: '1px solid var(--border, #333)',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                }}>
                  <span>Previous chats ({chatLogs.length})</span>
                  <button className="btn btn-secondary btn-icon" onClick={() => setShowHistory(false)} title="Close">
                    <ChevronLeft size={14} />
                  </button>
                </div>
                {chatLogs.length === 0 ? (
                  <div style={{ padding: 12, fontSize: '0.8rem', opacity: 0.7 }}>No saved chats yet.</div>
                ) : (
                  chatLogs.map(log => (
                    <div key={log.filename} style={{
                      display: 'flex', alignItems: 'center', gap: 8,
                      padding: '6px 12px', borderBottom: '1px solid var(--border-subtle, #222)',
                    }}>
                      <button
                        onClick={() => loadChat(log.filename)}
                        style={{ flex: 1, textAlign: 'left', background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', overflow: 'hidden' }}
                        title="Load this chat"
                      >
                        <div style={{ fontSize: '0.85rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {log.title || '(untitled)'}
                        </div>
                        <div style={{ fontSize: '0.7rem', opacity: 0.6 }}>
                          {log.timestamp?.slice(0, 19).replace('T', ' ')} · {log.message_count} msgs
                        </div>
                      </button>
                      <button className="btn btn-secondary btn-icon" onClick={() => deleteChatLog(log.filename)} title="Delete this log">
                        <Trash2 size={14} />
                      </button>
                    </div>
                  ))
                )}
              </div>
            )}
            {messages.length === 0 ? (
              <div className="empty-state">
                <Eye className="empty-state-icon" />
                <h3 className="empty-state-title">Welcome to Vision Lab</h3>
                <p className="empty-state-text">
                  Start a conversation, or upload an image or video for visual analysis. Media is optional.
                </p>
              </div>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}${msg.autonomous ? ' autonomous' : ''}`}>
                  <div className="message-content">
                    {msg.role === 'assistant' ? (() => {
                      const raw = msg.content || '';
                      const thinkMatch = raw.match(/^([\s\S]*?<\/think>)([\s\S]*)$/i);
                      const thinkBlock = thinkMatch ? thinkMatch[1] : null;
                      const displayContent = thinkMatch ? thinkMatch[2].trim() : raw;
                      return (
                        <>
                          {showObservation && msg.observation && (
                            <div className="thinking-block" style={{ borderLeftColor: 'var(--accent-secondary, #4a90e2)' }}>
                              <div className="thinking-block-header">
                                <Eye size={12} /> Observation (silent context for the response below)
                              </div>
                              <div className="thinking-block-content">{msg.observation}</div>
                            </div>
                          )}
                          {showThinking && thinkBlock && (
                            <div className="thinking-block">
                              <div className="thinking-block-header">
                                <Brain size={12} /> Thinking
                              </div>
                              <div className="thinking-block-content">{thinkBlock.replace(/<\/?think>/gi, '').trim()}</div>
                            </div>
                          )}
                          <ReactMarkdown
                            components={{
                              code({ className, children, ...props }) {
                                const match = /language-(\w+)/.exec(className || '');
                                const inline = !match;
                                return !inline ? (
                                  <SyntaxHighlighter
                                    style={oneDark}
                                    language={match[1]}
                                    PreTag="div"
                                  >
                                    {String(children).replace(/\n$/, '')}
                                  </SyntaxHighlighter>
                                ) : (
                                  <code className={className} {...props}>
                                    {children}
                                  </code>
                                );
                              }
                            }}
                          >
                            {displayContent}
                          </ReactMarkdown>
                        </>
                      );
                    })() : (
                      msg.content
                    )}
                  </div>
                  {(msg.hasMedia || msg.screenshot || msg.role === 'assistant') && (
                    <div className="message-meta">
                      {msg.hasMedia && (
                        <span className="message-attachment">
                          <Paperclip size={12} /> Media attached
                        </span>
                      )}
                      {msg.screenshot && (
                        <span className="message-attachment" style={{ cursor: 'pointer' }}
                          onClick={() => window.open(msg.screenshot, '_blank')}
                          title="Click to view full screenshot"
                        >
                          <Camera size={12} /> Screenshot attached
                        </span>
                      )}
                      {msg.autonomous && (
                        <span className="message-attachment" title="The model spoke unprompted, from your live screen">
                          <Monitor size={12} /> Live · unprompted
                        </span>
                      )}
                      {msg.role === 'assistant' && msg.content && (
                        ttsPlayingIdx === i ? (
                          <span className="message-attachment" style={{ cursor: 'pointer' }}
                            onClick={stopTTS}
                            title="Stop playback"
                          >
                            <Square size={12} /> Stop
                          </span>
                        ) : ttsLoadingIdx === i ? (
                          <span className="message-attachment" title="Synthesizing…">
                            <Volume2 size={12} /> …
                          </span>
                        ) : (
                          <span className="message-attachment" style={{ cursor: 'pointer' }}
                            onClick={() => speakText(msg.content, i)}
                            title="Speak this message via TTS"
                          >
                            <Volume2 size={12} /> Speak
                          </span>
                        )
                      )}
                    </div>
                  )}
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Chat Input */}
          <div className="chat-input-container">
            <div className="chat-input-wrapper">
              <textarea
                className="chat-input"
                placeholder="Type your message... (Enter to send)"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                disabled={isLoading}
              />
              {pendingScreenshot && (
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 8, padding: '6px 12px',
                  background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-accent)', marginBottom: 8,
                }}>
                  <img src={pendingScreenshot} alt="Screenshot" style={{
                    height: 48, borderRadius: 4, border: '1px solid var(--border-medium)',
                  }} />
                  <span style={{ fontSize: '0.8rem', color: 'var(--accent-primary)' }}>
                    <Camera size={14} /> Screenshot attached
                  </span>
                  <button
                    className="btn btn-secondary btn-icon"
                    onClick={clearScreenshot}
                    style={{ marginLeft: 'auto', padding: 4 }}
                    title="Remove screenshot"
                  >
                    <X size={14} />
                  </button>
                </div>
              )}
              <div className="chat-actions">
                <div className="chat-toggles">
                  <button
                    className={`context-toggle ${pendingScreenshot ? 'active' : ''}`}
                    onClick={captureScreenshot}
                    title="Capture left pane screenshot and attach to next message"
                  >
                    <Camera size={12} /> Screenshot
                  </button>
                  <button
                    className={`context-toggle ${liveSharing ? 'active' : ''}`}
                    onClick={liveSharing ? stopLiveShare : startLiveShare}
                    title={liveSharing ? 'Stop live screen sharing' : 'Share your screen live — the model observes it continuously'}
                  >
                    {liveSharing ? <><Square size={12} /> Stop Live</> : <><Monitor size={12} /> Live Screen</>}
                  </button>
                  <button
                    className={`context-toggle ${enablePaneContext ? 'active' : ''}`}
                    onClick={() => setEnablePaneContext(p => !p)}
                    title="Include left pane context in chat messages"
                  >
                    <Brain size={12} /> {enablePaneContext ? 'Context ON' : 'Context'}
                  </button>
                  <button
                    className={`context-toggle ${enableThinking ? 'active' : ''}`}
                    onClick={() => setEnableThinking(t => !t)}
                    title="Toggle Qwen3.5 thinking mode"
                  >
                    {enableThinking ? '🧠 Qwen3.5' : '⚡ Instruct'}
                  </button>
                  <button
                    className={`context-toggle ${enableTools ? 'active' : ''}`}
                    onClick={() => setEnableTools(t => !t)}
                    title="Enable agentic tool use (read/write files, list directories, web search, fetch URL)"
                  >
                    <Zap size={12} /> {enableTools ? 'Tools ON' : 'Tools'}
                  </button>
                  <button
                    className={`context-toggle ${showThinking ? 'active' : ''}`}
                    onClick={() => setShowThinking(t => !t)}
                    title="Show thinking/reasoning text in chat messages"
                  >
                    <Brain size={12} /> {showThinking ? 'Thinking Visible' : 'Show Thinking'}
                  </button>
                  <button
                    className={`context-toggle ${enableTTS ? 'active' : ''}`}
                    onClick={() => {
                      const next = !enableTTS;
                      setEnableTTS(next);
                      if (!next) stopTTS();
                    }}
                    title="Auto-speak assistant replies via the TTS server (configure backend tts_* keys in config.json)"
                  >
                    {enableTTS ? <Volume2 size={12} /> : <VolumeX size={12} />}
                    {enableTTS ? ' TTS ON' : ' TTS'}
                  </button>
                  <button
                    className={`context-toggle ${enableObservationPass ? 'active' : ''}`}
                    onClick={() => setEnableObservationPass(o => !o)}
                    title="When media is attached, run a structured observation pass first; inject result as silent context for the response"
                  >
                    <Eye size={12} /> {enableObservationPass ? 'Observe ON' : 'Observe'}
                  </button>
                  <button
                    className={`context-toggle ${showObservation ? 'active' : ''}`}
                    onClick={() => setShowObservation(o => !o)}
                    title="Show/hide the observation block above responses (when an Observe pass ran)"
                  >
                    {showObservation ? <Eye size={12} /> : <EyeOff size={12} />} {showObservation ? 'Obs Visible' : 'Show Obs'}
                  </button>
                </div>
                <div className="chat-controls">
                {media && media.id !== lastSentMedia && (
                  <div style={{
                    padding: '4px 8px',
                    background: 'var(--accent-glow)',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.75rem',
                    color: 'var(--accent-primary)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4
                  }}>
                    <Paperclip size={12} /> Ready
                  </div>
                )}
                <button
                  className="btn btn-secondary btn-icon"
                  onClick={handleRetry}
                  disabled={isLoading || messages.length < 2}
                  title="Retry last response"
                >
                  <RotateCcw size={16} />
                </button>
                <button
                  className={`btn btn-secondary btn-icon ${showHistory ? 'active' : ''}`}
                  onClick={toggleHistory}
                  title="Load a previous chat"
                >
                  <FolderOpen size={16} />
                </button>
                <button
                  className="btn btn-secondary btn-icon"
                  onClick={exportChat}
                  disabled={messages.length === 0}
                  title="Export chat as Markdown"
                >
                  <Download size={16} />
                </button>
                <button
                  className="btn btn-secondary btn-icon"
                  onClick={exportChatAsJSON}
                  disabled={messages.length === 0}
                  title="Export as JSON (REVEAL corpus format)"
                >
                  <FileText size={16} />
                </button>
                <button
                  className="btn btn-secondary btn-icon"
                  onClick={clearChat}
                  title="Clear chat"
                >
                  <Trash2 size={16} />
                </button>
                {isLoading ? (
                  <button
                    className="btn btn-danger"
                    onClick={handleStop}
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    className="btn btn-primary"
                    onClick={handleSend}
                    disabled={!input.trim()}
                  >
                    <Send size={16} /> Send
                  </button>
                )}
                </div>{/* end chat-controls */}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
