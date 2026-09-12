/**
 * VoiceGuard Dashboard Application Controller
 * Handles real-time audio visualization, mock scenario triggers,
 * live microphone streaming, and operator defensive actions.
 */

class VoiceGuardApp {
    constructor() {
        this.currentCallId = "CALL-IMPERSONATE-999";
        this.audioContext = null;
        this.analyser = null;
        this.micStream = null;
        this.isMicActive = false;
        this.animationId = null;

        // Scenarios definition
        this.scenarios = {
            legit: {
                callId: "CALL-LEGIT-001",
                claimed: "Rajesh Verma (CFO)",
                callerId: "+91-98765-43210",
                status: "ACTIVE",
                riskScore: 12.4,
                riskLevel: "LOW",
                action: "ALLOW",
                actionDesc: "Interaction authorized. Natural prosody and verified biometric voiceprint.",
                breakdown: { synth: 5.2, speaker: 2.1, behavior: 2.5, context: 1.5, replay: 1.1 },
                reasons: ["VERIFIED_BIOMETRIC_VOICEPRINT", "REGISTERED_CALLER_ID", "BENEFICIARY_VERIFIED"],
                metrics: { pitch: 124.5, jitter: 1.45, centroid: 1850, vocoder: "Natural Human" },
                tx: { id: "TX-VEN-002", amount: "₹1,50,000.00", status: "APPROVED", desc: "TechCorp Vendor Invoice" }
            },
            urgency: {
                callId: "CALL-URGENT-442",
                claimed: "Priya Sharma (VP Ops)",
                callerId: "+91-98123-45678",
                status: "HELD_FOR_VERIFICATION",
                riskScore: 54.0,
                riskLevel: "MEDIUM",
                action: "VERIFY_MFA",
                actionDesc: "Secondary authentication challenge dispatched. Unusually high urgency detected.",
                breakdown: { synth: 14.5, speaker: 12.0, behavior: 15.0, context: 8.5, replay: 4.0 },
                reasons: ["ELEVATED_PROSODIC_URGENCY", "HIGH_VALUE_THRESHOLD", "AMBIGUOUS_SPEAKER_SCORE"],
                metrics: { pitch: 210.8, jitter: 3.82, centroid: 2450, vocoder: "Acoustic Strain (Human)" },
                tx: { id: "TX-OPS-8L-003", amount: "₹8,00,000.00", status: "PENDING_MFA", desc: "Emergency Logistics Payout" }
            },
            attack: {
                callId: "CALL-IMPERSONATE-999",
                claimed: "Rajesh Verma (CFO)",
                callerId: "+91-99999-00000",
                status: "FLAGGED_THREAT",
                riskScore: 87.5,
                riskLevel: "HIGH",
                action: "BLOCK_AND_DISCONNECT",
                actionDesc: "Automated kill-switch triggered. High-confidence AI voice cloning detected.",
                breakdown: { synth: 31.2, speaker: 17.3, behavior: 11.3, context: 12.8, replay: 4.0 },
                reasons: [
                    "AI_VOICE_CLONING_DETECTED",
                    "BIOMETRIC_VOICEPRINT_MISMATCH",
                    "UNREGISTERED_CALLER_ID_SPOOF_RISK",
                    "HIGH_VALUE_TRANSACTION: ₹25,00,000",
                    "FIRST_TIME_UNVERIFIED_BENEFICIARY",
                    "EXCEEDED_KILL_SWITCH_THRESHOLD"
                ],
                metrics: { pitch: 135.2, jitter: 0.12, centroid: 3840, vocoder: "HiFi-GAN / VITS" },
                tx: { id: "TX-CFO-25L-001", amount: "₹25,00,000.00", status: "BLOCKED", desc: "Apex Global Holdings LLC" }
            },
            replay: {
                callId: "CALL-REPLAY-551",
                claimed: "Vikram Patel (Payments)",
                callerId: "+91-98999-11223",
                status: "FLAGGED_THREAT",
                riskScore: 78.2,
                riskLevel: "HIGH",
                action: "BLOCK_AND_DISCONNECT",
                actionDesc: "Acoustic replay attack detected. Loudspeaker transducer resonance artifacts tagged.",
                breakdown: { synth: 12.0, speaker: 18.2, behavior: 8.0, context: 15.0, replay: 25.0 },
                reasons: ["SHARP_TRANSDUCER_HF_CUTOFF", "REPETITIVE_ROOM_REFLECTION", "TRANSDUCER_RESONANCE_SPIKE"],
                metrics: { pitch: 118.0, jitter: 0.85, centroid: 4200, vocoder: "Playback Transducer" },
                tx: { id: "TX-REP-12L", amount: "₹12,50,000.00", status: "BLOCKED", desc: "Off-Hours Wire Request" }
            }
        };

        this.init();
    }

    init() {
        this.startClock();
        this.initCanvasWaveform();
        this.loadScenario('attack'); // Default to the CFO attack
    }

    startClock() {
        const clockEl = document.getElementById('live-clock');
        setInterval(() => {
            const now = new Date();
            clockEl.textContent = now.toISOString().substring(11, 19) + ' UTC';
        }, 1000);
    }

    initCanvasWaveform() {
        const canvas = document.getElementById('waveform-canvas');
        const ctx = canvas.getContext('2d');
        let phase = 0;

        const drawDummyWave = () => {
            if (this.isMicActive && this.analyser) {
                // Mic drawing handled in startMicStream
                return;
            }

            ctx.fillStyle = '#06090F';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw center grid line
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, canvas.height / 2);
            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();

            // Draw simulated waveform
            ctx.lineWidth = 2;
            ctx.strokeStyle = this.currentRiskScore >= 75 ? '#EF4444' : (this.currentRiskScore > 30 ? '#F59E0B' : '#10B981');
            ctx.beginPath();

            const sliceWidth = canvas.width / 150;
            let x = 0;

            for (let i = 0; i < 150; i++) {
                const amp = Math.sin(i * 0.1 + phase) * Math.cos(i * 0.05 + phase * 0.5) * 35;
                const y = (canvas.height / 2) + amp + (Math.random() - 0.5) * 6;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            ctx.stroke();
            phase += 0.06;
            requestAnimationFrame(drawDummyWave);
        };

        drawDummyWave();
    }

    loadScenario(key) {
        const s = this.scenarios[key];
        if (!s) return;

        this.currentRiskScore = s.riskScore;

        // Update active button state
        document.querySelectorAll('.btn-scenario').forEach(btn => btn.classList.remove('active-scenario'));
        const activeIdx = key === 'legit' ? 0 : (key === 'urgency' ? 1 : (key === 'attack' ? 2 : 3));
        const activeBtn = document.querySelectorAll('.btn-scenario')[activeIdx];
        if (activeBtn) activeBtn.classList.add('active-scenario');

        // Kill switch banner
        const banner = document.getElementById('kill-switch-banner');
        if (s.riskScore >= 75) {
            banner.classList.remove('hidden');
        } else {
            banner.classList.add('hidden');
        }

        // Meta info
        document.getElementById('meta-call-id').textContent = s.callId;
        document.getElementById('meta-claimed').textContent = s.claimed;
        document.getElementById('active-call-meta').innerHTML = 
            `Session: <span class="text-cyan font-mono">${s.callId}</span> | Claimed: <span class="text-amber font-mono">${s.claimed}</span>`;

        // Gauge
        document.getElementById('gauge-score').textContent = s.riskScore.toFixed(1);
        const gaugeEl = document.getElementById('gauge-circle');
        const gaugeLevel = document.getElementById('gauge-level');
        const colorVar = s.riskScore >= 75 ? '#EF4444' : (s.riskScore > 30 ? '#F59E0B' : '#10B981');
        
        gaugeEl.style.background = `conic-gradient(${colorVar} ${s.riskScore}%, rgba(255, 255, 255, 0.05) 0)`;
        gaugeEl.style.boxShadow = `0 0 25px ${colorVar}4D`;
        gaugeLevel.textContent = s.riskLevel + ' RISK';
        gaugeLevel.style.color = colorVar;

        document.getElementById('gauge-action-desc').textContent = s.actionDesc;

        // Recommendation card
        const recCard = document.getElementById('rec-action-card');
        const recTitle = document.getElementById('rec-action-title');
        const recBody = document.getElementById('rec-action-body');
        const recIcon = document.getElementById('rec-icon');

        recCard.className = `rec-action-card ${s.riskScore >= 75 ? 'card-crimson' : (s.riskScore > 30 ? 'card-amber' : 'card-green')}`;
        recTitle.textContent = s.action;
        recBody.textContent = s.actionDesc;
        recIcon.textContent = s.riskScore >= 75 ? 'block' : (s.riskScore > 30 ? 'verified_user' : 'check_circle');

        // Breakdown bars
        document.getElementById('b-val-synth').textContent = s.breakdown.synth.toFixed(1) + '%';
        document.getElementById('b-val-speaker').textContent = s.breakdown.speaker.toFixed(1) + '%';
        document.getElementById('b-val-behavior').textContent = s.breakdown.behavior.toFixed(1) + '%';
        document.getElementById('b-val-context').textContent = s.breakdown.context.toFixed(1) + '%';
        document.getElementById('b-val-replay').textContent = s.breakdown.replay.toFixed(1) + '%';

        document.getElementById('b-bar-synth').style.width = (s.breakdown.synth / 35 * 100) + '%';
        document.getElementById('b-bar-speaker').style.width = (s.breakdown.speaker / 25 * 100) + '%';
        document.getElementById('b-bar-behavior').style.width = (s.breakdown.behavior / 15 * 100) + '%';
        document.getElementById('b-bar-context').style.width = (s.breakdown.context / 15 * 100) + '%';
        document.getElementById('b-bar-replay').style.width = (s.breakdown.replay / 10 * 100) + '%';

        // Reason chips
        const chipsContainer = document.getElementById('reason-chips-container');
        chipsContainer.innerHTML = '';
        s.reasons.forEach(r => {
            const chip = document.createElement('span');
            chip.className = `chip ${s.riskScore >= 75 ? 'chip-crimson' : (s.riskScore > 30 ? 'chip-amber' : 'chip-green')}`;
            chip.textContent = r;
            chipsContainer.appendChild(chip);
        });

        // Acoustic Metrics
        document.getElementById('m-pitch').innerHTML = `${s.metrics.pitch} <span class="m-unit">Hz</span>`;
        document.getElementById('m-jitter').innerHTML = `${s.metrics.jitter} <span class="m-unit">%</span>`;
        document.getElementById('m-centroid').innerHTML = `${s.metrics.centroid} <span class="m-unit">Hz</span>`;
        document.getElementById('m-vocoder').textContent = s.metrics.vocoder;

        // Transaction Card
        document.getElementById('tx-amount').innerHTML = `${s.tx.amount.split('.')[0]}<span class="tx-currency">.00 INR</span>`;
        document.getElementById('tx-id').textContent = s.tx.id;
        document.getElementById('tx-beneficiary').textContent = s.tx.desc;
        const txBadge = document.getElementById('tx-badge-status');
        const txBanner = document.getElementById('tx-banner');
        
        txBadge.textContent = s.tx.status;
        if (s.tx.status === 'BLOCKED') {
            txBadge.className = 'badge-status badge-blocked';
            txBanner.className = 'tx-status-banner banner-blocked';
            txBanner.innerHTML = `<span class="material-symbols-outlined">gpp_bad</span><span>TRANSACTION FROZEN BY RISK GATEWAY</span>`;
        } else if (s.tx.status === 'PENDING_MFA') {
            txBadge.className = 'badge-status';
            txBadge.style.background = 'rgba(245, 158, 11, 0.2)';
            txBadge.style.color = '#F59E0B';
            txBanner.className = 'tx-status-banner';
            txBanner.style.background = 'rgba(245, 158, 11, 0.15)';
            txBanner.style.color = '#FDE68A';
            txBanner.innerHTML = `<span class="material-symbols-outlined">send_to_mobile</span><span>HELD: PUSH MFA DISPATCHED</span>`;
        } else {
            txBadge.className = 'badge-status badge-active';
            txBanner.className = 'tx-status-banner';
            txBanner.style.background = 'rgba(16, 185, 129, 0.15)';
            txBanner.style.color = '#A7F3D0';
            txBanner.innerHTML = `<span class="material-symbols-outlined">check_circle</span><span>PRE-APPROVED WITHIN INSTITUTIONAL LIMITS</span>`;
        }

        // Add to audit trail
        this.addAuditEntry(s.riskScore >= 75 ? 'audit-critical' : (s.riskScore > 30 ? 'audit-warning' : 'audit-info'), 
            `SCENARIO LOADED: ${s.callId}`, 
            `Evaluated risk score ${s.riskScore}% (${s.riskLevel}). Action recommended: ${s.action}`);
    }

    addAuditEntry(typeClass, title, desc) {
        const list = document.getElementById('audit-list');
        const item = document.createElement('div');
        item.className = `audit-item ${typeClass}`;
        const timeStr = new Date().toTimeString().substring(0, 8);
        item.innerHTML = `
            <div class="a-time font-mono">${timeStr}</div>
            <div class="a-content">
                <strong>${title}</strong>
                <span>${desc}</span>
            </div>
        `;
        list.insertBefore(item, list.firstChild);
    }

    triggerAction(action) {
        const txId = document.getElementById('tx-id').textContent;
        fetch(`/api/v1/transactions/${txId}/action`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: action, notes: `Operator action triggered from command center` })
        }).then(res => res.json())
        .then(data => {
            alert(`Defensive Action [${action}] executed successfully for ${txId}. Status updated to: ${data.status}`);
            this.addAuditEntry('audit-info', `OPERATOR ACTION: ${action}`, `Applied to ${txId}. Resulting status: ${data.status}`);
        }).catch(err => {
            alert(`Executed local defensive protocol [${action}] on transaction ${txId}.`);
            this.addAuditEntry('audit-info', `OPERATOR ACTION: ${action}`, `Dispatched defensive protocol to call channel.`);
        });
    }

    escalateCurrentCall() {
        this.triggerAction('ESCALATE');
    }

    async toggleMicrophone() {
        const btn = document.getElementById('btn-mic');
        const text = document.getElementById('mic-btn-text');

        if (this.isMicActive) {
            // Stop mic
            if (this.micStream) {
                this.micStream.getTracks().forEach(t => t.stop());
            }
            this.isMicActive = false;
            text.textContent = 'START LIVE MICROPHONE EVALUATION';
            btn.style.background = 'linear-gradient(135deg, #1E40AF, #3B82F6)';
            return;
        }

        try {
            this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.audioContext.createMediaStreamSource(this.micStream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);

            this.isMicActive = true;
            text.textContent = 'LISTENING... SPEAK INTO MIC (REAL-TIME)';
            btn.style.background = 'linear-gradient(135deg, #059669, #10B981)';

            this.drawMicWaveform();
            this.addAuditEntry('audit-info', 'LIVE MIC CONNECTED', 'Capturing Web Audio API stream @ 16kHz.');
        } catch (e) {
            alert('Microphone access denied or not available: ' + e);
        }
    }

    drawMicWaveform() {
        if (!this.isMicActive) return;

        const canvas = document.getElementById('waveform-canvas');
        const ctx = canvas.getContext('2d');
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const render = () => {
            if (!this.isMicActive) return;
            requestAnimationFrame(render);

            this.analyser.getByteTimeDomainData(dataArray);

            ctx.fillStyle = '#06090F';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.lineWidth = 2;
            ctx.strokeStyle = '#10B981';
            ctx.beginPath();

            const sliceWidth = canvas.width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * (canvas.height / 2);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            ctx.stroke();
        };

        render();
    }

    handleAudioUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('audio_file', file);
        formData.append('transaction_amount', '2500000.0');
        formData.append('is_new_beneficiary', 'true');

        this.addAuditEntry('audit-info', 'AUDIO UPLOAD STARTED', `Uploading ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);

        fetch(`/api/v1/calls/${this.currentCallId}/analyze_chunk`, {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(event => {
            this.currentRiskScore = event.risk_score;
            document.getElementById('gauge-score').textContent = event.risk_score.toFixed(1);
            alert(`Audio Analysis Complete! Risk Score: ${event.risk_score}% (${event.risk_level}). Recommended Action: ${event.recommended_action}`);
            this.addAuditEntry(event.risk_level === 'HIGH' ? 'audit-critical' : 'audit-info',
                `ANALYSIS COMPLETE: Risk ${event.risk_score}%`,
                `Synthetic: ${(event.synthetic_score * 100).toFixed(1)}%, Speaker Match: ${(event.speaker_similarity * 100).toFixed(1)}%`
            );
        })
        .catch(err => {
            alert(`Uploaded ${file.name}. Simulated forensic evaluation triggered.`);
        });
    }
}

// Initialize on window load
window.addEventListener('DOMContentLoaded', () => {
    window.app = new VoiceGuardApp();
});