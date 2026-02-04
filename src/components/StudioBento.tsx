import { BentoGrid, BentoCard } from "@/components/ui/BentoGrid";
import {
    Search,
    Database,
    Workflow,
    Fingerprint,
    FileAudio,
    Maximize2,
    Video
} from "lucide-react";

// Map studio IDs to Lucide icons
const getIconForStudio = (id: string) => {
    switch (id) {
        case 'live-search': return Search;
        case 'db-search': return Database;
        case 'logic-flow': return Workflow;
        case 'audio-fingerprint': return Fingerprint;
        case 'optic-to-audio': return FileAudio;
        case 'set-extender': return Maximize2;
        case 'video-sync': return Video;
        default: return Search;
    }
};

interface StudioRaw {
    id: string;
    data: {
        name: string;
        description: string;
        accent?: string;
        // other fields if needed
    }
}

export function StudioBento({ studios }: { studios: StudioRaw[] }) {
    return (
        <BentoGrid className="lg:grid-cols-3">
            {studios.map((studio) => {
                const IconComponent = getIconForStudio(studio.id);
                const bgClass = studio.data.accent
                    ? `bg-gradient-to-br from-${studio.data.accent}/5 to-transparent`
                    : "bg-gradient-to-br from-neutral-200/50 to-neutral-100/50 dark:from-neutral-900/50 dark:to-neutral-800/50";

                return (
                    <BentoCard
                        key={studio.id}
                        name={studio.data.name}
                        className={studio.id === 'db-search' || studio.id === 'live-search' ? 'col-span-3 lg:col-span-2' : 'col-span-3 lg:col-span-1'}
                        Icon={IconComponent}
                        description={studio.data.description}
                        href={`/vibeset/studios/${studio.id}`}
                        cta="Enter Studio"
                        backgroundClass={bgClass}
                    />
                );
            })}
        </BentoGrid>
    );
}
