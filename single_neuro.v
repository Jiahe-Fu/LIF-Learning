module LIF_neuro(
    parameter signed [15:0] V_rest = 16'd17920,  // 静息电位 (0.7V)
    parameter signed [15:0] V_th = 16'd15360,   // 阈值电位 (0.6V)
    parameter signed [15:0] V_reset = 16'd17920, // 重置电位 (0.7V)
    parameter tau_m = 15,   // 膜时间常数 (10ms)
    parameter t_ref = 2,     //不应期
    parameter dt_ms = 100,
    parameter SCALE = 1 << 8,
    parameter signed [15:0] dt_fixed = 16'd26, // 将时间步长转换为固定点表示,25.6四舍五入
    parameter signed [15:0] tau_m_fixed = 16'd3840 // 将膜时间常数转换为固定点表示
) (
    input wire clk,
    input wire rst_n,
    input wire signed [15:0] input_current, // 上游神经元的输入电流
    output reg spike // 输出脉冲
)
reg signed [15:0] V; // 膜电位
// reg signed [15:0] spike;
reg [4:0] ref_counter; // 不应期计数器
wire if_ref; // 是否处于不应期
assign if_ref = (ref_counter > 0);  //ref_counter大于0表示处于不应期

wire [31:0] signed a;// 计算的变量，这里可能会溢出，所以要32位
wire [31:0] signed dV;// dV, 为了和上面的计算对齐
wire [16:0] dV_fixed; // dV的定点表示

assign a = V_rest - V + $signed({16'd0 , input_current})
assign dV = (a * dt_fixed + (tau_m_fixed >> 1)) / tau_m_fixed;
assign dV_fixed = dV[15:0];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        V <= V_rest; // 初始化膜电位
        spike <= 1'd0; // 初始化脉冲输出
        ref_counter <= 5'd0; // 初始化不应期计数器

    end
    else begin
        if (if_ref)
        begin
            ref_counter <= ref_counter - 1; // 不应期计数器递减
            spike <= 1'd0; // 不应期内不产生脉冲
        end
        else
        begin
            // 膜电位更新逻辑

//这里可以优化            V <= V + ((-V + input_current) * dt_fixed) / tau_m_fixed;
            V <= V + dV_fixed;
            if (V >= V_th) begin
                spike <= 1'd1; // 产生脉冲
                V <= V_reset; // 重置膜电位
                ref_counter <= t_ref; // 设置不应期
            end
            else begin
                spike <= 1'd0; // 未达到阈值，不产生脉冲
            end
        end
    end
    
end


endmodule