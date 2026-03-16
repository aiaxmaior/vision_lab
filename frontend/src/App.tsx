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
import html2canvas from 'html2canvas';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  Upload, Settings, Cpu, MessageSquare, Send, Trash2,
  ChevronDown, ChevronRight, RefreshCw, Paperclip, X,
  Scissors, Check, Zap, Eye, Brain, Film, Camera,
  FolderOpen, Save, ChevronLeft, FileText, ImageIcon, Download, RotateCcw, RotateCw, EyeOff
} from 'lucide-react';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  hasMedia?: boolean;
  screenshot?: string;
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
  { value: "<think>{content}</think>", label: "Qwen3.5 (Default)" },
  { value: "", label: "No special tag" }
];

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
  const [activeTab, setActiveTab] = useState<'captions' | 'batch' | 'prompts' | 'batch-review'>('batch');
  const [enablePaneContext, setEnablePaneContext] = useState(false);
  const [enableThinking, setEnableThinking] = useState(true);
  const [enableTools, setEnableTools] = useState(false);
  const [showThinking, setShowThinking] = useState(false);
  const [showCaptionPreview, setShowCaptionPreview] = useState(true);

  // Prompt Manager state
  const [promptProfiles, setPromptProfiles] = useState<{ filename: string; name: string; is_default: boolean; description: string }[]>([]);
  const [activeProfile, setActiveProfile] = useState('prompts.json');
  const [promptData, setPromptData] = useState<any>(null);
  const [promptEditSection, setPromptEditSection] = useState<string>('chat_assistant');
  const [promptEditKey, setPromptEditKey] = useState<string>('system_prompt'); { }
  const [promptEditValue, setPromptEditValue] = useState('');
  const [promptSaveStatus, setPromptSaveStatus] = useState('');
  const [newProfileName, setNewProfileName] = useState('');

  // Screenshot capture state
  const [pendingScreenshot, setPendingScreenshot] = useState<string | null>(null);
  const functionPaneRef = useRef<HTMLDivElement>(null);

  // Batch captioner state
  const [batchDir, setBatchDir] = useState('');
  const [batchInstruction, setBatchInstruction] = useState('');
  const [captionTarget, setCaptionTarget] = useState('general');
  const [captionTargets, setCaptionTargets] = useState<{ id: string; name: string; description: string; style: string; token_limit: number | null; media_types: string[] }[]>([]);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchLog, setBatchLog] = useState<BatchLogEntry[]>([]);
  const [batchProgress, setBatchProgress] = useState({ completed: 0, total: 0, skipped: 0, errors: 0 });
  const batchLogEndRef = useRef<HTMLDivElement>(null);

  // Caption reviewer state
  const [captionDir, setCaptionDir] = useState('');
  const [captionPairs, setCaptionPairs] = useState<CaptionPair[]>([]);
  const [captionIndex, setCaptionIndex] = useState(0);
  const [captionEdit, setCaptionEdit] = useState('');
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

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

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
          interaction_mode: config.interaction_mode,
          system_prompt: config.system_prompt,
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
          pane_context: getPaneContext() || undefined,
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
              if (parsed.tool_call) {
                const tc = parsed.tool_call;
                const argsStr = JSON.stringify(tc.arguments, null, 2);
                assistantContent += `\n\n**🔧 Tool Call:** \`${tc.name}\`\n\`\`\`json\n${argsStr}\n\`\`\`\n`;
                setMessages(prev => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1] = { role: 'assistant', content: assistantContent };
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
                      : tr.name === 'list_directory'
                        ? `✅ Found ${tr.result.count} entries in \`${tr.result.path}\``
                        : `✅ ${JSON.stringify(tr.result)}`;
                assistantContent += `**Result:** ${resStr}\n\n`;
                setMessages(prev => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1] = { role: 'assistant', content: assistantContent };
                  return newMsgs;
                });
              } else if (parsed.content) {
                assistantContent += parsed.content;
                setMessages(prev => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1] = { role: 'assistant', content: assistantContent };
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
      abortControllerRef.current = null;
      setMessages(prev => { autoSaveChat(prev); return prev; });
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    const includeMedia = !!(media && media.id !== lastSentMedia);
    const hasScreenshot = !!pendingScreenshot;

    setMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      hasMedia: includeMedia,
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
    try {
      await fetch('/api/chat/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: msgs, title })
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
    setIsLoading(false);
  };

  const clearChat = () => {
    setMessages([]);
    setLastSentMedia(null);
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
      const res = await fetch(`/api/captions/scan?directory=${encodeURIComponent(dir)}`);
      if (!res.ok) throw new Error('Directory not found');
      const data = await res.json();
      setCaptionPairs(data.pairs);
      setCaptionIndex(0);
      if (data.pairs.length > 0) {
        setCaptionEdit(data.pairs[0].caption);
      }
    } catch (e) {
      setCaptionPairs([]);
      setCaptionIndex(0);
      setCaptionEdit('');
    } finally {
      setCaptionLoading(false);
    }
  };

  const navigateCaption = (direction: number) => {
    if (captionPairs.length === 0) return;
    const newIndex = (captionIndex + direction + captionPairs.length) % captionPairs.length;
    setCaptionIndex(newIndex);
    setCaptionEdit(captionPairs[newIndex].caption);
    setCaptionSaveStatus('');
  };

  const rerunCaption = async () => {
    if (captionPairs.length === 0 || rerunning) return;
    const pair = captionPairs[captionIndex];
    setRerunning(true);
    setCaptionSaveStatus('');
    setCaptionEdit('');
    let accumulated = '';
    try {
      const res = await fetch('/api/captions/rerun', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: pair.image_path,
          existing_caption: pair.caption,
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
              setCaptionEdit(accumulated);
            } else if (msg.done) {
              setCaptionEdit(msg.caption);
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

  const saveCaptionEdit = async () => {
    if (captionPairs.length === 0) return;
    const pair = captionPairs[captionIndex];
    try {
      const res = await fetch('/api/captions/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ caption_path: pair.caption_path, caption: captionEdit })
      });
      if (res.ok) {
        setCaptionSaveStatus('Saved ✓');
        setCaptionPairs(prev => {
          const updated = [...prev];
          updated[captionIndex] = { ...updated[captionIndex], caption: captionEdit.trim(), has_caption: true };
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
    try {
      const res = await fetch('/api/captions/delete', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ caption_path: pair.caption_path, image_path: pair.image_path })
      });
      if (res.ok) {
        setCaptionPairs(prev => {
          const updated = [...prev];
          updated.splice(captionIndex, 1);
          return updated;
        });
        setCaptionIndex(prev => Math.max(0, Math.min(prev, captionPairs.length - 2)));
        setCaptionEdit('');
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

  const getPaneContext = (): string => {
    if (!enablePaneContext) return '';
    if (activeTab === 'batch') {
      const { completed, total, skipped, errors } = batchProgress;
      const lastDone = [...batchLog].reverse().find(e => e.type === 'done');
      const targetName = captionTargets.find(t => t.id === captionTarget)?.name || captionTarget;
      return `[Batch Captioner] Target: ${targetName}, Directory: ${batchDir || 'not set'}, Progress: ${completed}/${total} (${skipped} skipped, ${errors} errors), Status: ${batchRunning ? 'running' : 'idle'}${lastDone ? `, Last caption: "${lastDone.caption_preview?.slice(0, 100)}..."` : ''}`;
    }
    if (activeTab === 'captions' && currentPair) {
      return `[Caption Reviewer] File: ${currentPair.filename} (${captionIndex + 1}/${captionPairs.length}), Has caption: ${currentPair.has_caption}, Current edit: "${captionEdit.slice(0, 200)}${captionEdit.length > 200 ? '...' : ''}"`;
    }
    return '';
  };

  const captureScreenshot = async () => {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
      const track = stream.getVideoTracks()[0];
      const imageCapture = new ImageCapture(track);
      const bitmap = await imageCapture.grabFrame();
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

  const currentPair = captionPairs[captionIndex] || null;
  const captionStats = {
    total: captionPairs.length,
    withCaptions: captionPairs.filter(p => p.has_caption).length,
    charCount: captionEdit.length,
    wordCount: captionEdit.trim() ? captionEdit.trim().split(/\s+/).length : 0,
    tokenCount: Math.ceil(captionEdit.length / 4),
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
                      value={config.thought_syntax}
                      onChange={(e) => setConfig(c => ({ ...c, thought_syntax: e.target.value }))}
                    >
                      {THOUGHT_SYNTAXES.map(s => (
                        <option key={s.value} value={s.value}>{s.label}</option>
                      ))}
                    </select>
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
                  <label htmlFor="batch-thinking" className="form-label" style={{ margin: 0 }}>Enable Thinking</label>
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
                    placeholder="Directory path (e.g., /path/to/dataset)"
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
                {captionPairs.length > 0 && (
                  <div className="caption-review-stats">
                    <span>{captionStats.total} images</span>
                    <span>{captionStats.withCaptions} captioned</span>
                    {captionStats.total - captionStats.withCaptions > 0 && (
                      <span className="caption-missing">
                        {captionStats.total - captionStats.withCaptions} missing
                      </span>
                    )}
                  </div>
                )}
              </div>

              {captionPairs.length === 0 ? (
                <div className="empty-state">
                  <ImageIcon className="empty-state-icon" />
                  <h3 className="empty-state-title">Caption Reviewer</h3>
                  <p className="empty-state-text">
                    Enter a directory path containing images and .txt caption files to review and edit them.
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
                      style={{
                        width: '100%',
                        marginBottom: 8,
                        fontSize: '0.75rem',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: 8
                      }}
                    >
                      {showCaptionPreview ? <EyeOff size={14} /> : <Eye size={14} />}
                      {showCaptionPreview ? 'Hide Preview' : 'Show Preview'}
                    </button>
                    <div className="caption-review-filename">
                      {currentPair?.filename}
                      {currentPair && !currentPair.has_caption && (
                        <span className="caption-missing-badge">No caption file</span>
                      )}
                    </div>
                  </div>

                  <div className="caption-review-edit-panel">
                    <textarea
                      className="caption-review-textarea"
                      value={captionEdit}
                      onChange={(e) => { setCaptionEdit(e.target.value); setCaptionSaveStatus(''); }}
                      placeholder="Enter caption text..."
                    />
                    <div className="caption-rerun-row">
                      <input
                        type="text"
                        className="form-input"
                        placeholder="Extra instructions for re-caption (optional)..."
                        value={rerunInstruction}
                        onChange={(e) => setRerunInstruction(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter') rerunCaption(); }}
                        disabled={rerunning}
                        style={{ flex: 1, fontSize: '0.8rem' }}
                      />
                      <button
                        className={`btn btn-secondary ${rerunning ? 'loading' : ''}`}
                        onClick={rerunCaption}
                        disabled={rerunning}
                        title="Re-caption this image using the current model"
                      >
                        <RefreshCw size={14} /> {rerunning ? 'Running…' : 'Re-caption'}
                      </button>
                    </div>
                    <div className="caption-review-edit-footer">
                      <div className="caption-review-counts">
                        <span>{captionStats.charCount} chars</span>
                        <span>~{captionStats.wordCount} words</span>
                        <span>~{captionStats.tokenCount} tokens</span>
                      </div>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        {captionSaveStatus && (
                          <span className="caption-save-status">{captionSaveStatus}</span>
                        )}
                        <button
                          className="btn btn-danger"
                          onClick={deleteCaption}
                          disabled={!currentPair}
                          title="Delete image and caption file"
                        >
                          <Trash2 size={16} /> Delete
                        </button>
                        <button className="btn btn-primary" onClick={saveCaptionEdit}>
                          <Save size={16} /> Save
                        </button>
                      </div>
                    </div>
                  </div>
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
                      <div className="batch-review-grid">
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
                                <RotateCcw size={10} />
                              </button>
                              <button
                                className="batch-review-card-btn"
                                onClick={() => rotateBatchImage(si, ii, -90)}
                                title="Rotate CW"
                              >
                                <RotateCw size={10} />
                              </button>
                              <button
                                className="batch-review-card-delete"
                                onClick={() => deleteBatchImage(si, ii)}
                                title="Delete image and caption"
                              >
                                <Trash2 size={10} />
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
          ) : null}
        </div>

        {/* Pane Resize Handle */}
        <div
          className={`resize-handle ${isPaneResizing ? 'active' : ''}`}
          onMouseDown={handlePaneMouseDown}
        />

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
                <div key={i} className={`message ${msg.role}`}>
                  <div className="message-content">
                    {msg.role === 'assistant' ? (() => {
                      const raw = msg.content || '';
                      const thinkMatch = raw.match(/^([\s\S]*?<\/think>)([\s\S]*)$/i);
                      const thinkBlock = thinkMatch ? thinkMatch[1] : null;
                      const displayContent = thinkMatch ? thinkMatch[2].trim() : raw;
                      return (
                        <>
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
                  {(msg.hasMedia || msg.screenshot) && (
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
                    {enableThinking ? '🧠 Thinking' : '⚡ Instruct'}
                  </button>
                  <button
                    className={`context-toggle ${enableTools ? 'active' : ''}`}
                    onClick={() => setEnableTools(t => !t)}
                    title="Enable agentic tool use (read/write files, list directories)"
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
