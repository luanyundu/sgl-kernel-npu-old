#include <memory>

#include "exception.hpp"
#include "pytorch_npu_helper.hpp"
namespace deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(c10::DeviceType::PrivateUse1);
        event->record(c10_npu::getCurrentNPUStream());
    }

    explicit EventHandle(const c10_npu::NPUStream& stream) {
        event = std::make_shared<torch::Event>(c10::DeviceType::PrivateUse1);
        event->record(stream);
    }

    EventHandle(const EventHandle &other) = default;

    void current_stream_wait() const
    {
        c10_npu::getCurrentNPUStream().unwrap().wait(*event);
    }
};

torch::Event create_event(const c10_npu::NPUStream& s);
void stream_wait(const c10_npu::NPUStream& s_0, const c10_npu::NPUStream& s_1);
void stream_wait(const c10_npu::NPUStream& s, const EventHandle& event);

}  // namespace deep_ep
