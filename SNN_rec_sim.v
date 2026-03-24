// 这里fpga的浮点数运算走的很慢，所以需要浮点转换为定点，那么我像以往一样，取Q8.8，即16位数，前8位为小数点前，后八位为小数点后
// 先写编码函数，这里依旧使用最简单的rate coding编码，即用在时间T内所发出的脉冲数来表示信息
module rate_coding(
    parameter PIXEL_W = 8,  // 像素位宽
    parameter T = 30  // 时间T
) (
    input clk,
    input rst_n,
    input [PIXEL_W - 1 : 0] pixel,    // 输入的一个像素点的亮度，取值范围为0-128
    input start,                    // 控制rating模块的调用
    output out_spike
)
    reg [15 : 0] lfsr;
    reg [5 : 0] cnt;

    always @(posedge clk or negedge rst_n)  begin
        if (!rst_n) begin
            lfsr <= 16'hACE1;   // 给定一个初始种子，保证可重复性，方便debug
            cnt <= 0;
            out_spike <= 0;
        end
        else if (start and cnt < T)
        begin
            lfsr <= {lfsr[14:0], lfsr[15]^lfsr[13]^lfsr[12]^lfsr[10]};
            if (pixel < lfsr[7:0])
            begin
                out_spike <= 1;
            end
            else begin 
                out_spike <= 0;
            end
            cnt <= cnt + 1;
        end

    end
endmodule